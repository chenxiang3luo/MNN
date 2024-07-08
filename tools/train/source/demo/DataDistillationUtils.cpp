//
//  MobilenetV2Utils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
// cd .. && cd build/ && ./runTrainDemo.out DataDistillationTrain /root/datasets/Mnist
#include "DataDistillationUtils.hpp"
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <MNN/expr/ExprCreator.hpp>
#include <vector>
#include "DataLoader.hpp"
#include "ClassDataLoader.hpp"
#include "DemoUnit.hpp"
#include "NN.hpp"
#include "SGD.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include <MNN/MNNDefine.h>
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "MnistDataset.hpp"
#include "Transformer.hpp"
#include "ImageDataset.hpp"
#include "module/PipelineModule.hpp"
#include "MobilenetV2NoBN.hpp"
#include "ConvNet.hpp"


#include <random>
using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace MNN::CV;
using namespace MNN::Train::Model;


cv::Mat varpToMat(VARP var) {
    // Assume var is already in NHWC format
    var = _Squeeze(var,{0});
    auto info = var->getInfo();
    auto ptr = var->readMap<float>();
    int height = info->dim[0];
    int width = info->dim[1];
    int channels = info->dim[2];

    cv::Mat mat(height, width, CV_32FC(channels), const_cast<float*>(ptr));

    // Normalize to 0-255 and convert to 8-bit unsigned
    cv::Mat norm_mat;
    cv::normalize(mat, norm_mat, 0, 255, cv::NORM_MINMAX);
    norm_mat.convertTo(norm_mat, CV_8UC(channels));

    return norm_mat;
}

void save_picture(VARP var, bool is_batch,bool is_init, int class_num) {
    // Assume the data is in NHWC format
    auto tmp = _Cast<float>(var);
    tmp = _Convert(tmp, NHWC);
    auto info = tmp->getInfo();
    int batch_size = info->dim[0];
    INTS split_size(batch_size,1);
    auto var_single = _Split(tmp, split_size, 0);
    std::string folderPath;
    if (is_init){
        folderPath = "/root/datasets/DD_datasets/real/" + std::to_string(class_num);
        std::string command;
        command = "mkdir -p " + folderPath;
        system(command.c_str());
    } else {
        folderPath = "/root/datasets/DD_datasets/syn/" + std::to_string(class_num);
        std::string command;
        command = "mkdir -p " + folderPath;
        system(command.c_str());
    }
    
    if (is_batch) {
        for (int i = 0; i < batch_size; ++i) {
            std::string filename;
            if (is_init){
                filename = folderPath + "/" + std::to_string(i) + ".jpg";
            } else{
                filename = folderPath + "/" + std::to_string(i) + ".jpg";
            }
            cv::Mat mat = varpToMat(var_single[i]);
            
            cv::imwrite(filename, mat);
            std::cout << "Saved " << filename << std::endl;
        }
    } else {
        cv::Mat mat = varpToMat(tmp);
        cv::imwrite("vapr_output.png", mat);
        std::cout << "Saved vapr_output.png" << std::endl;
    }
}



void DataDistillationUtils::train(std::string model_name, const int numClasses, const int addToLabel,
                                std::string root, const int quantBits) {
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);

    

    
    // solver->setMomentum2(0.99f);
    // solver->setWeightDecay(0.00004f);

    auto dataset = MnistDataset::create(root, MnistDataset::Mode::TRAIN);
    size_t dataset_size = dataset.mDataset->size();
    std::cout << " dataset_size " << dataset_size << std::endl;
    const size_t batchSize  = 128;
    const size_t numWorkers = 0;
    const int ipc = 1000;
    bool shuffle            = true;

    
    std::vector<std::vector<size_t>> indices_class(numClasses);
    std::shared_ptr<MNN::Train::MnistDataset> mmdataset = std::static_pointer_cast<MnistDataset>(dataset.mDataset);
    VARP labels = mmdataset->labels();
    // labels->getTensor()->print();
    
    auto data = labels->readMap<uint8_t>();
    // printVector(test1);
    // printVector(test2);
    // // 根据标签将索引添加到 indices_class 中
    for (size_t i = 0; i < dataset_size; ++i) {
        uint8_t lab = data[i];
        // printf("%d ", data[i]);
        indices_class[lab].push_back(i);
    }
    std::cout<<indices_class[0].size()<<std::endl;

    
    // dataset.mDataset,batchSize,indices_class, true, shuffle, numWorkers
    auto trainDataLoader = std::shared_ptr<ClassDataLoader>(ClassDataLoader::makeDataLoader(dataset.mDataset,batchSize,indices_class,true,shuffle,numWorkers));
    std::vector<int> shapeValue = {10,1,28,28};
    
    VARP random_shape = _Const(shapeValue.data(), {4}, NHWC, halide_type_of<int>());
    VARPS syn_inputs;
    auto init_method = "real";
    if(init_method == "random"){
        for (int c = 0; c < numClasses; c++){
            auto syn_input = _RandomUnifom(random_shape, halide_type_of<float>(),0.0f, 1.0f,c,c);
            syn_input.fix(VARP::TRAINABLE);
            // syn_input->getTensor()->print();
            syn_inputs.push_back(syn_input);
            // std::cout<<syn_inputs.size()<<std::endl;
            save_picture(syn_input,true,true,c);
        }
    }else if (init_method == "real")
    {
        for (int c = 0; c < numClasses; c++){
            trainDataLoader->select_class(c);
            auto example  = trainDataLoader->next(ipc)[0];
            auto tpm = _Cast<float>(example.first[0]) * _Const(1.0f / 255.0f);
            auto syn_input = _Cast<float>(tpm);
            syn_input.fix(VARP::TRAINABLE);
            // syn_input->getTensor()->print();
            syn_inputs.push_back(syn_input);
            // std::cout<<syn_inputs.size()<<std::endl;
            save_picture(syn_input,true,true,c);
        }
        
        
    }else{
        printf("unsopported init methods");
    }
    
    // return;
    std::cout << " syn " << syn_inputs.size() << std::endl;
    std::shared_ptr<Module> _input(Module::createEmpty(syn_inputs));

    std::shared_ptr<SGD> solver(new SGD(_input));
    solver->setMomentum(0.5f);
    solver->setLearningRate(1.f);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    

    size_t trainIterations = 1000;
    // size_t trainIterations = trainDataLoader->iterNumber();
    // auto testDataset            = MnistDataset::create(root, MnistDataset::Mode::TEST);
    // const size_t testBatchSize  = 20;
    // const size_t testNumWorkers = 0;
    // shuffle                     = false;

    // auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    // printVector(size);
    // const int usedSize = 1000;
    // const int testIterations = usedSize / testBatchSize;
    ;
    for (int epoch = 0; epoch < 1; ++epoch) {
        // model->clearCache();
        exe->gc(Executor::FULL);
        {
            // AUTOTIME;
            
            
            
            for (int i = 0; i < trainIterations; i++) {
                AUTOTIME;
                VARP loss = _Const(0.f,{1},NCHW);
                std::shared_ptr<MNN::Train::Model::ConvNet> model(new ConvNet(10,1));
                model->clearCache();
                model->setIsTraining(true);
                bool use_BN = true;
                if (use_BN){
                    VARPS real_inputs;
                    for (uint8_t c = 0; c < numClasses; c++){
                        
                        trainDataLoader->select_class(c);
                        auto example  = trainDataLoader->next(batchSize)[0];
                        auto syn_input = syn_inputs[c];
                        // std::cout << " 2" << std::endl;
                        // std::cout <<example.first[0]->getInfo()->order<<std::endl;
                        // printVector(example.first[0]->getInfo()->dim);
                        // std::cout <<syn_input->getInfo()->order<<std::endl;
                        auto cast1      = _Cast<float>(example.first[0]);
                        example.first[0] = cast1 * _Const(1.0f / 255.0f);
                        real_inputs.emplace_back(example.first[0]);
                        // syn_input = syn_input * _Const(1.0f / 255.0f);
                        
                        // auto loss    = _CrossEntropy(predict, newTarget);
                        // float rate   = LrScheduler::inv(0.0001, solver->currentStep(), 0.0001, 0.75);
                        // loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                        
                    }
                    auto real_feature = model->embedding(_Convert(_Concat(real_inputs,0), NCHW));
                    auto syn_feature = model->embedding(_Convert(_Concat(syn_inputs,0), NCHW));
                    loss = loss + _ReduceSum(_Square(_ReduceMean(_Reshape(real_feature, {numClasses,batchSize, -1}),{1}) - _ReduceMean(_Reshape(syn_feature, {numClasses,ipc, -1}),{1})));
                }else {
                    for (uint8_t c = 0; c < numClasses; c++){
                    
                        trainDataLoader->select_class(c);
                        auto example  = trainDataLoader->next(batchSize)[0];
                        auto syn_input = syn_inputs[c];
                        // std::cout << " 2" << std::endl;
                        // std::cout <<example.first[0]->getInfo()->order<<std::endl;
                        // printVector(example.first[0]->getInfo()->dim);
                        // std::cout <<syn_input->getInfo()->order<<std::endl;
                        auto cast1      = _Cast<float>(example.first[0]);
                        example.first[0] = cast1 * _Const(1.0f / 255.0f);
                        // syn_input = syn_input * _Const(1.0f / 255.0f);
                        auto real_feature = model->embedding(_Convert(example.first[0], NCHW));
                        auto syn_feature = model->embedding(_Convert(syn_input, NCHW));
                        // auto loss    = _CrossEntropy(predict, newTarget);
                        // float rate   = LrScheduler::inv(0.0001, solver->currentStep(), 0.0001, 0.75);
                        // loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                        loss = loss + _ReduceSum(_Square(_ReduceMean(real_feature,{0}) - _ReduceMean(syn_feature,{0})));
                    }
                }
                
                // Compute One-Hot
                // auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(addToLabel), {})),
                //                   _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                //                          _Scalar<float>(0.0f));

                if (solver->currentStep() % 10 == 0) {
                    std::cout << "train iteration: " << solver->currentStep();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    // std::cout << " lr: " << rate << std::endl;
                }
                solver->step(loss);
                // std::cout << "order " << syn_inputs[0]->getInfo()->order<< std::endl;
                //test
            
                
                // imwrite(image_name,syn_inputs[0]);
            }
            for (int c = 0; c < numClasses; c++){
                save_picture(syn_inputs[c],true,false,c);
            }
            
        }
        // syn_inputs[0]->imwrite;
        // int correct = 0;
        // int sampleCount = 0;
        // testDataLoader->reset();
        // model->setIsTraining(false);
        // exe->gc(Executor::PART);

        // AUTOTIME;
        // for (int i = 0; i < testIterations; i++) {
        //     auto data       = testDataLoader->next();
        //     auto example    = data[0];
        //     auto predict    = model->forward(_Convert(example.first[0], NC4HW4));
        //     predict         = _ArgMax(predict, 1); // (N, numClasses) --> (N)
        //     auto label = _Squeeze(example.second[0]) + _Scalar<int32_t>(addToLabel);
        //     sampleCount += label->getInfo()->size;
        //     auto accu       = _Cast<int32_t>(_Equal(predict, label).sum({}));
        //     correct += accu->readMap<int32_t>()[0];

        //     if ((i + 1) % 10 == 0) {
        //         std::cout << "test iteration: " << (i + 1) << " ";
        //         std::cout << "acc: " << correct << "/" << sampleCount << " = " << float(correct) / sampleCount * 100 << "%";
        //         std::cout << std::endl;
        //     }
        // }
        // auto accu = (float)correct / testDataLoader->size();
        // // auto accu = (float)correct / usedSize;
        // std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;

        // {
        //     auto forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
        //     forwardInput->setName("data");
        //     auto predict = model->forward(forwardInput);
        //     Transformer::turnModelToInfer()->onExecute({predict});
        //     predict->setName("prob");
        //     std::string fileName = "temp.mobilenetv2.mnn";
        //     Variable::save({predict}, fileName.c_str());

        // }
    }
}
// def get_images(c, n): # get random n images from class c
//             idx_shuffle = np.random.permutation(indices_class[c])[:n]
//             return images_all[idx_shuffle]