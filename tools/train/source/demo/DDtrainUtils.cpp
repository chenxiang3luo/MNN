//
//  DDtrainUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
// cd .. && cd build/ && ./runTrainDemo.out DDtrain /root/datasets/DD_datasets/real /root/datasets/DD_datasets/dataset.txt /root/datasets/Mnist

#include "DDtrainUtils.hpp"
#include <MNN/expr/Executor.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include "MnistDataset.hpp"
#include "ImageDataset.hpp"
#include "RandomSampler.hpp"
#include "NN.hpp"
#include "SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "ConvNet.hpp"
#include "MobilenetV2.hpp"
#include <opencv2/opencv.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace MNN::Train::Model;


cv::Mat varpToM(VARP var) {
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

void save_tmp(VARP var, bool is_batch,bool is_init, int class_num) {
    // Assume the data is in NHWC format
    auto tmp = _Cast<float>(var);
    tmp = _Convert(tmp, NHWC);
    auto info = tmp->getInfo();
    int batch_size = info->dim[0];
    INTS split_size(batch_size,1);
    auto var_single = _Split(tmp, split_size, 0);
    // std::string folderPath;
    // if (is_init){
    //     folderPath = "/root/datasets/DD_datasets/real/" + std::to_string(class_num);
    //     std::string command;
    //     command = "mkdir -p " + folderPath;
    //     system(command.c_str());
    // } else {
    //     folderPath = "/root/datasets/DD_datasets/syn/" + std::to_string(class_num);
    //     std::string command;
    //     command = "mkdir -p " + folderPath;
    //     system(command.c_str());
    // }
    
    if (is_batch) {
        for (int i = 0; i < batch_size; ++i) {
            std::string filename;
            if (is_init){
                filename =  "./" + std::to_string(i) + ".png";
            } else{
                filename = "./" + std::to_string(i) + ".png";
            }
            cv::Mat mat = varpToM(var_single[i]);
            
            cv::imwrite(filename, mat);
            std::cout << "Saved " << filename << std::endl;
        }
    } else {
        cv::Mat mat = varpToM(tmp);
        cv::imwrite("vapr_output.png", mat);
        std::cout << "Saved vapr_output.png" << std::endl;
    }
}

void DDtrainUtils::train(std::string model_name, std::string syn_root, std::string syn_pathToImageTxt, std::string test_root) {
    // {
    //     // Load snapshot
    //     auto para = Variable::load("mnist.snapshot.mnn");
    //     model->loadParameters(para);
    // }
    std::shared_ptr<MobilenetV2> model(new MobilenetV2(10,1));
    int numClasses = 10;
    auto exe = Executor::getGlobalExecutor();
    BackendConfig back_config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, back_config, 4);
//    _initTensorStatic();
    std::shared_ptr<SGD> sgd(new SGD(model));
    sgd->setMomentum(0.9f);
    sgd->setLearningRate(0.01f);
    // sgd->setMomentum2(0.99f);
    sgd->setWeightDecay(0.0005f);
    

    
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    
    
    // // use ImageDataset
    auto converImagesToFormat  = CV::GRAY;
    int resizeHeight           = 28;
    int resizeWidth            = 28;
    // std::vector<float> scales = {1/255.0, 1/255.0, 1/255.0};
    
    std::shared_ptr<ImageDataset::ImageConfig> config(ImageDataset::ImageConfig::create(converImagesToFormat));
    bool readAllImagesToMemory = false;
    auto dataset = ImageDataset::create(syn_root, syn_pathToImageTxt, config.get(), readAllImagesToMemory);
    
    const size_t batchSize  = 128;
    const size_t numWorkers = 1;
    bool shuffle            = true;
    auto dataLoader = dataset.createLoader(batchSize, true, false, numWorkers);
    size_t iterations = dataLoader->iterNumber();



    // use MnistDataset
    // auto dataset = MnistDataset::create(test_root, MnistDataset::Mode::TRAIN);
    // const size_t batchSize  = 64;
    // const size_t numWorkers = 0;
    // bool shuffle            = true;
    // auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));
    // size_t iterations = dataLoader->iterNumber();
    
    auto testDataset            = MnistDataset::create(test_root, MnistDataset::Mode::TEST);
    const size_t testBatchSize  = 64;
    const size_t testNumWorkers = 0;
    shuffle                     = false;

    auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    size_t testIterations = testDataLoader->iterNumber();
    int epoches = 300;
    int lr_schedule = epoches/2;


    for (int epoch = 0; epoch < epoches; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;
            int lastIndex = 0;
            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                // auto cast       = _Cast<float>(example.first[0]);
                auto cast       = _Cast(example.first[0], halide_type_of<float>());
                example.first[0] = _Multiply(cast, _Const(1.0f / 255.0f));
                // example.first[0] = cast;
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                
                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(1), {})),
                                  _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

                auto predict = model->forward(_Convert(example.first[0], NCHW));

                
                auto loss = _CrossEntropy(predict, newTarget);
    
                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
                    std::cout << "epoch: " << (epoch);
                    std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << (i - lastIndex) <<  " iter"  << std::endl;
                    std::cout.flush();
                    _100Time.reset();
                    lastIndex = i;
                }

                sgd->step(loss);

            }
        }
        if (epoch==lr_schedule) {
            sgd->setMomentum(0.9f);
            // sgd->setMomentum2(0.99f);
            sgd->setWeightDecay(0.0005f);
            sgd->setLearningRate(0.01 * 0.1f);
        }
        // Variable::save(model->parameters(), "mnist.snapshot.mnn");
        // {
        //     model->setIsTraining(false);
        //     auto forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
        //     forwardInput->setName("data");
        //     auto predict = model->forward(forwardInput);
        //     predict->setName("prob");
        //     Transformer::turnModelToInfer()->onExecute({predict});
        //     Variable::save({predict}, "temp.mnist.mnn");
        // }

        int correct = 0;
        testDataLoader->reset();
        model->setIsTraining(false);
        int moveBatchSize = 0;
        for (int i = 0; i < testIterations; i++) {
            auto data       = testDataLoader->next();
            auto example    = data[0];
            moveBatchSize += example.first[0]->getInfo()->dim[0];
            if ((i + 1) % 100 == 0) {
                std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
            }
            auto cast       = _Cast<float>(example.first[0]);
            // example.first[0] = cast * _Const(1.0f / 255.0f);
            example.first[0] = cast * _Const(1.0f / 255.0f);
            auto predict    = model->forward(example.first[0]);
            predict         = _ArgMax(predict, 1);
            
            auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.second[0]))).sum({});
            correct += accu->readMap<int32_t>()[0];
        }
        auto accu = (float)correct / (float)testDataLoader->size();
        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
    }
}
