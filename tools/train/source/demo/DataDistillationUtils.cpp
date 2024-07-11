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
#include "LambdaTransform.hpp"
#include "StackTransform.hpp"
#include "Transform.hpp"
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

VARP vectorToVARP(const std::vector<int>& vec) {
    // 获取vector的大小
    int size = vec.size();

    // 使用_CONST函数创建VARP
    VARP var = _Const(vec.data(), {size}, NCHW, halide_type_of<int>());
    
    return var;
}

VARP _AffineGrid(VARP theta, int N, int C, int H, int W) {
    // Create normalized 2D grid
    auto linspace_x = _LinSpace(_Scalar<int32_t>(-1), _Scalar<int32_t>(1), _Scalar<int32_t>(W));
    auto linspace_y = _LinSpace(_Scalar<int32_t>(-1), _Scalar<int32_t>(1), _Scalar<int32_t>(H));

    auto grid_x = _Tile(_Unsqueeze(linspace_x, {0}), vectorToVARP({H, 1}));
    auto grid_y = _Tile(_Unsqueeze(linspace_y, {1}), vectorToVARP({1, W}));
    auto grid = _Stack({grid_x, grid_y}, 2);  // [H, W, 2]

    // Add homogeneous coordinate
    auto ones = _Const(1.0f, {H, W, 1}, NHWC);
    grid = _Concat({grid, ones}, 2);  // [H, W, 3]

    // Reshape to [1, H*W, 3] and repeat N times
    grid = _Reshape(grid, {1, H * W, 3});
    grid = _Tile(grid, vectorToVARP({N, 1, 1}));  // [N, H*W, 3]

    // Apply affine transformation
    auto theta_reshaped = _Reshape(theta, {N, 2, 3});  // [N, 2, 3]
    auto grid_transformed = _BatchMatMul(theta_reshaped, grid, false, true);  // [N, 2, H*W]
    grid_transformed = _Reshape(grid_transformed, {N, H, W, 2});  // [N, H, W, 2]

    return grid_transformed;
}



cv::Mat clamp(const cv::Mat& src, double min_val, double max_val) {
    cv::Mat dst;
    cv::max(src, min_val, dst); // 将小于 min_val 的值设为 min_val
    cv::min(dst, max_val, dst); // 将大于 max_val 的值设为 max_val
    return dst;
}

VARP rand_scale(VARP x, float ratio) {
    // float ratio = param.ratio_scale;
    
    std::vector<float> sx(x->getInfo()->dim[0]);
    std::vector<float> sy(x->getInfo()->dim[0]);
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        sx[i] = ((float)rand() / RAND_MAX) * (ratio - 1.0f/ratio) + 1.0f/ratio;
        sy[i] = ((float)rand() / RAND_MAX) * (ratio - 1.0f/ratio) + 1.0f/ratio;
    }

    std::vector<std::vector<std::vector<float>>> theta(x->getInfo()->dim[0], std::vector<std::vector<float>>(2, std::vector<float>(3)));
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        theta[i][0][0] = sx[i];
        theta[i][0][1] = 0;
        theta[i][0][2] = 0;
        theta[i][1][0] = 0;
        theta[i][1][1] = sy[i];
        theta[i][1][2] = 0;
    }

    // if (param.Siamese) {
    //     for (int i = 1; i < x->getInfo()->dim[0]; ++i) {
    //         theta[i] = theta[0];
    //     }
    // }

    auto theta_var = _Const(theta.data(), {x->getInfo()->dim[0], 2, 3}, NHWC, halide_type_of<float>());
    auto grid = _AffineGrid(theta_var,x->getInfo()->dim[0], x->getInfo()->dim[1], x->getInfo()->dim[2], x->getInfo()->dim[3]);
    auto result = _GridSample(x, grid);
    return result;
}

VARP rand_rotate(VARP x, float ratio) {
    // float ratio = param.ratio_rotate;
    // set_seed_DiffAug(param);
    
    std::vector<float> theta(x->getInfo()->dim[0]);
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        theta[i] = (((float)rand() / RAND_MAX) - 0.5f) * 2 * ratio / 180 * M_PI;
    }

    std::vector<std::vector<std::vector<float>>> affine(x->getInfo()->dim[0], std::vector<std::vector<float>>(2, std::vector<float>(3)));
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        affine[i][0][0] = cos(theta[i]);
        affine[i][0][1] = -sin(theta[i]);
        affine[i][0][2] = 0;
        affine[i][1][0] = sin(theta[i]);
        affine[i][1][1] = cos(theta[i]);
        affine[i][1][2] = 0;
    }

    // if (param.Siamese) {
    //     for (int i = 1; i < x->getInfo()->dim[0]; ++i) {
    //         affine[i] = affine[0];
    //     }
    // }

    auto affine_var = _Const(affine.data(), {x->getInfo()->dim[0], 2, 3}, NHWC, halide_type_of<float>());
    auto grid = _AffineGrid(affine_var,x->getInfo()->dim[0], x->getInfo()->dim[1], x->getInfo()->dim[2], x->getInfo()->dim[3]);
    auto result = _GridSample(x, grid);
    return result;
}

VARP rand_flip(VARP x, float prob) {
    // float prob = param.prob_flip;
    // set_seed_DiffAug(param);

    std::vector<float> randf(x->getInfo()->dim[0]);
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        randf[i] = (float)rand() / RAND_MAX;
    }

    // if (param.Siamese) {
    //     for (int i = 1; i < x->getInfo()->dim[0]; ++i) {
    //         randf[i] = randf[0];
    //     }
    // }

    auto randf_var = _Const(randf.data(), {x->getInfo()->dim[0], 1, 1, 1}, NHWC, halide_type_of<float>());
    auto mask = _Greater(randf_var, _Const(prob, randf_var->getInfo()->dim, randf_var->getInfo()->order));
    auto flipped_x = _Reverse(x, _Scalar<float>(3));
    auto result = _Select(mask, flipped_x, x);
    return result;
}

VARP rand_brightness(VARP x,float ratio) {
    // float ratio = param.brightness;
    // set_seed_DiffAug(param);

    std::vector<float> randb(x->getInfo()->dim[0]);
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        randb[i] = (float)rand() / RAND_MAX;
    }

    // if (param.Siamese) {
    //     for (int i = 1; i < x->getInfo()->dim[0]; ++i) {
    //         randb[i] = randb[0];
    //     }
    // }

    auto randb_var = _Const(randb.data(), {x->getInfo()->dim[0], 1, 1, 1}, NHWC, halide_type_of<float>());
    auto adjusted = x + (randb_var - _Scalar<float>(.5f)) * _Scalar<float>(ratio);
    return adjusted;
}

VARP rand_saturation(VARP x, float ratio) {
    // float ratio = param.saturation;
    auto x_mean = _ReduceMean(x, {1}, true);
    // set_seed_DiffAug(param);

    std::vector<float> rands(x->getInfo()->dim[0]);
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        rands[i] = (float)rand() / RAND_MAX;
    }

    // if (param.Siamese) {
    //     for (int i = 1; i < x->getInfo()->dim[0]; ++i) {
    //         rands[i] = rands[0];
    //     }
    // }

    auto rands_var = _Const(rands.data(), {x->getInfo()->dim[0], 1, 1, 1}, NHWC, halide_type_of<float>());
    auto adjusted = (x - x_mean) * (rands_var * _Scalar<float>(ratio)) + x_mean;
    return adjusted;
}

VARP rand_contrast(VARP x, float ratio) {
    // float ratio = param.contrast;
    auto x_mean = _ReduceMean(x, {1, 2, 3}, true);
    // set_seed_DiffAug(param);

    std::vector<float> randc(x->getInfo()->dim[0]);
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        randc[i] = (float)rand() / RAND_MAX;
    }

    // if (param.Siamese) {
    //     for (int i = 1; i < x->getInfo()->dim[0]; ++i) {
    //         randc[i] = randc[0];
    //     }
    // }

    auto randc_var = _Const(randc.data(), {x->getInfo()->dim[0], 1, 1, 1}, NHWC, halide_type_of<float>());
    auto adjusted = (x - x_mean) * (randc_var + _Scalar<float>(ratio)) + x_mean;
    return adjusted;
}

VARP createMeshGrid(int batch_size, int height, int width) {
    // 创建x轴坐标
    auto linspace_x = _LinSpace(0, _Scalar<int>(width - 1), _Scalar<int>(width));
    auto meshgrid_x = _Tile(_Unsqueeze(linspace_x, {0}), vectorToVARP({height, 1}));
    
    // 创建y轴坐标
    auto linspace_y = _LinSpace(0, _Scalar<int>(height - 1), _Scalar<int>(height));
    auto meshgrid_y = _Tile(_Unsqueeze(linspace_y, {1}), vectorToVARP({1, width}));
    
    // 合并x和y坐标
    auto grid_x = _Tile(meshgrid_x, vectorToVARP({batch_size, 1, 1}));
    auto grid_y = _Tile(meshgrid_y, vectorToVARP({batch_size, 1, 1}));
    auto grid = _Concat({grid_x, grid_y}, 1);  // [batch_size, 2, height, width]
    
    return grid;
}

// VARP rand_crop(VARP x, float ratio) {
//     // float ratio = param.ratio_crop_pad;
//     int shift_x = static_cast<int>(x->getInfo()->dim[2] * ratio + 0.5);
//     int shift_y = static_cast<int>(x->getInfo()->dim[3] * ratio + 0.5);
//     // set_seed_DiffAug(param);

//     std::vector<int> translation_x(x->getInfo()->dim[0]);
//     std::vector<int> translation_y(x->getInfo()->dim[0]);
//     for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
//         translation_x[i] = rand() % (2 * shift_x + 1) - shift_x;
//         translation_y[i] = rand() % (2 * shift_y + 1) - shift_y;
//     }

//     // if (param.Siamese) {
//     //     for (int i = 1; i < x->getInfo()->dim[0]; ++i) {
//     //         translation_x[i] = translation_x[0];
//     //         translation_y[i] = translation_y[0];
//     //     }
//     // }

//     auto translation_x_var = _Const(translation_x.data(), {x->getInfo()->dim[0], 1, 1}, NHWC, halide_type_of<int>());
//     auto translation_y_var = _Const(translation_y.data(), {x->getInfo()->dim[0], 1, 1}, NHWC, halide_type_of<int>());

//     auto x_pad = _Pad(x, vectorToVARP({1, 1, 1, 1, 0, 0, 0, 0}));
//     auto grid = createMeshGrid(x->getInfo()->dim[0], x->getInfo()->dim[2], x->getInfo()->dim[3]);
//     auto grid_x = clamp(grid[1] + translation_x_var + 1, 0, x->getInfo()->dim[2] + 1);
//     auto grid_y = clamp(grid[2] + translation_y_var + 1, 0, x->getInfo()->dim[3] + 1);

//     auto result = x_pad->permute({0, 2, 3, 1}).index({grid[0], grid_x, grid_y}).permute({0, 3, 1, 2});
//     return result;
// }


void printVector(const std::vector<int>& vec) {
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

Example mnistTransform(Example example) {
        // // an easier way to do this

        auto mean = 0.1307f; 
        auto std = 0.3081f;
        auto cast       = _Cast(example.first[0], halide_type_of<float>());
        example.first[0] = _Multiply(cast, _Const(1.0f / 255.0f));
        example.first[0] = example.first[0] - _Const(mean);
        example.first[0] = _Multiply(example.first[0],_Const(1.f/std));
        return example;
    }

cv::Mat varpToMat(VARP var) {
    // Assume var is already in NHWC format
    var = _Squeeze(var,{0});
    auto info = var->getInfo();
    auto ptr = var->readMap<float>();
    int height = info->dim[0];
    int width = info->dim[1];
    int channels = info->dim[2];

    cv::Mat mat(height, width, CV_32FC(channels), const_cast<float*>(ptr));
    std::vector<cv::Mat> imgChannels(channels);
    cv::split(mat,imgChannels);
    std::vector<float> means = {0.1307f}; 
    std::vector<float> stds = {0.3081f};
    for(int i = 0; i < channels;i++){
        imgChannels[0] = imgChannels[i] * stds[i] + means[i];
    }
    cv::Mat de_img;
    cv::merge(imgChannels,de_img);
    // mat.convertTo(de_img, CV_32FC(channels), std, mean)
    // Normalize to 0-255 and convert to 8-bit unsigned
    cv::Mat norm_mat;
    cv::normalize(clamp(de_img,0,1), norm_mat, 0, 255, cv::NORM_MINMAX);
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
                filename = folderPath + "/" + std::to_string(i) + ".png";
            } else{
                filename = folderPath + "/" + std::to_string(i) + ".png";
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
    const int ipc = 1;
    bool shuffle            = true;

    auto trainTransform = std::make_shared<LambdaTransform>(mnistTransform);
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

    std::vector<std::shared_ptr<BatchTransform>> transforms;
    transforms.emplace_back(trainTransform);
    transforms.emplace_back(std::shared_ptr<StackTransform>(new StackTransform));

    // dataset.mDataset,batchSize,indices_class, true, shuffle, numWorkers
    auto trainDataLoader = std::shared_ptr<ClassDataLoader>(ClassDataLoader::makeDataLoader(dataset.mDataset,transforms,batchSize,indices_class,shuffle,numWorkers));
    std::vector<int> shapeValue = {ipc,1,28,28};
    
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
            auto tpm = _Cast<float>(example.first[0]);
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
    

    size_t trainIterations = 10000;
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
                        // auto cast1      = _Cast<float>(example.first[0]);
                        // example.first[0] = cast1 * _Const(1.0f / 255.0f);
                        real_inputs.emplace_back(example.first[0]);
                        // syn_input = syn_input * _Const(1.0f / 255.0f);
                        
                        // auto loss    = _CrossEntropy(predict, newTarget);
                        // float rate   = LrScheduler::inv(0.0001, solver->currentStep(), 0.0001, 0.75);
                        // loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                        
                    }
                    auto real_feature = model->embedding(_Convert(_Concat(real_inputs,0), NCHW));
                    auto syn_feature = model->embedding(_Convert(_Concat(syn_inputs,0), NCHW));
                    // printVector(syn_feature->getInfo()->dim);
                    loss = loss + _ReduceSum(_Square(_ReduceMean(_Reshape(real_feature, {numClasses,batchSize, -1}),{1}) - _ReduceMean(_Reshape(syn_feature, {numClasses,ipc, -1}),{1})));
                }else {
                    for (uint8_t c = 0; c < numClasses; c++){
                    
                        trainDataLoader->select_class(c);
                        auto example  = trainDataLoader->next(batchSize)[0];
                        auto syn_input = syn_inputs[c];
                        // auto cast1      = _Cast<float>(example.first[0]);
                        // example.first[0] = cast1 * _Const(1.0f / 255.0f);
                        // syn_input = syn_input * _Const(1.0f / 255.0f);
                        auto real_feature = model->embedding(_Convert(example.first[0], NCHW));
                        auto syn_feature = model->embedding(_Convert(syn_input, NCHW));
                        // auto loss    = _CrossEntropy(predict, newTarget);
                        // float rate   = LrScheduler::inv(0.0001, solver->currentStep(), 0.0001, 0.75);
                        // loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                        loss = loss + _ReduceSum(_Square(_ReduceMean(real_feature,{0}) - _ReduceMean(syn_feature,{0})));
                    }
                }
                

                if (solver->currentStep() % 50 == 0) {
                    std::cout << "train iteration: " << solver->currentStep();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    for (int c = 0; c < numClasses; c++){
                        save_picture(syn_inputs[c],true,false,c);
                    }
                    // std::cout << " lr: " << rate << std::endl;
                }
                solver->step(loss);
                // std::cout << "order " << syn_inputs[0]->getInfo()->order<< std::endl;
                //test
            
                
                // imwrite(image_name,syn_inputs[0]);
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