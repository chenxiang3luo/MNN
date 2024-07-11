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
#include <experimental/random>
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

VARP clamp(VARP x, VARP min, VARP max) {
        return _Maximum(_Minimum(x, max), min);
    }

cv::Mat cv_clamp(const cv::Mat& src, double min_val, double max_val) {
    cv::Mat dst;
    cv::max(src, min_val, dst); // 将小于 min_val 的值设为 min_val
    cv::min(dst, max_val, dst); // 将大于 max_val 的值设为 max_val
    return dst;
}

VARP rand_scale(VARP x, float ratio_scale) {
    // float ratio = param.ratio_scale;
    
    std::vector<float> sx(x->getInfo()->dim[0]);
    std::vector<float> sy(x->getInfo()->dim[0]);
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        sx[i] = ((float)rand() / RAND_MAX) * (ratio_scale - 1.0f/ratio_scale) + 1.0f/ratio_scale;
        sy[i] = ((float)rand() / RAND_MAX) * (ratio_scale - 1.0f/ratio_scale) + 1.0f/ratio_scale;
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

VARP rand_rotate(VARP x, float ratio_rotate) {
    // float ratio = param.ratio_rotate;
    // set_seed_DiffAug(param);
    
    std::vector<float> theta(x->getInfo()->dim[0]);
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        theta[i] = (((float)rand() / RAND_MAX) - 0.5f) * 2 * ratio_rotate / 180 * M_PI;
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

VARP rand_flip(VARP x, float prob_flip) {
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
    auto mask = _Greater(randf_var, _Const(prob_flip, randf_var->getInfo()->dim, randf_var->getInfo()->order));
    auto flipped_x = _Reverse(x, _Scalar<float>(3));
    auto result = _Select(mask, flipped_x, x);
    return result;
}

VARP rand_brightness(VARP x,float brightness) {
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
    auto adjusted = x + (randb_var - _Scalar<float>(.5f)) * _Scalar<float>(brightness);
    return adjusted;
}

VARP rand_saturation(VARP x, float saturation) {
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
    auto adjusted = (x - x_mean) * (rands_var * _Scalar<float>(saturation)) + x_mean;
    return adjusted;
}

VARP rand_contrast(VARP x, float contrast) {
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
    auto adjusted = (x - x_mean) * (randc_var + _Scalar<float>(contrast)) + x_mean;
    return adjusted;
}


VARP rand_crop(VARP x, float ratio_crop_pad) {
    // float ratio = param.ratio_crop_pad;
    int shift_x = static_cast<int>(x->getInfo()->dim[2] * ratio_crop_pad + 0.5);
    int shift_y = static_cast<int>(x->getInfo()->dim[3] * ratio_crop_pad + 0.5);
    // set_seed_DiffAug(param);

    std::vector<int> translation_x(x->getInfo()->dim[0]);
    std::vector<int> translation_y(x->getInfo()->dim[0]);
    for (int i = 0; i < x->getInfo()->dim[0]; ++i) {
        translation_x[i] = rand() % (2 * shift_x + 1) - shift_x;
        translation_y[i] = rand() % (2 * shift_y + 1) - shift_y;
    }

    // if (param.Siamese) {
    //     for (int i = 1; i < x->getInfo()->dim[0]; ++i) {
    //         translation_x[i] = translation_x[0];
    //         translation_y[i] = translation_y[0];
    //     }
    // }
    int batch_size = x->getInfo()->dim[0];
    int height = x->getInfo()->dim[2];
    int width = x->getInfo()->dim[3];

    auto translation_x_var = _Const(translation_x.data(), {x->getInfo()->dim[0], 1, 1}, NHWC, halide_type_of<int>());
    auto translation_y_var = _Const(translation_y.data(), {x->getInfo()->dim[0], 1, 1}, NHWC, halide_type_of<int>());

    std::vector<int> batch_range(batch_size);
    std::vector<int> height_range(height);
    std::vector<int> width_range(width);
    std::iota(batch_range.begin(), batch_range.end(), 0);
    std::iota(height_range.begin(), height_range.end(), 0);
    std::iota(width_range.begin(), width_range.end(), 0);

    VARP grid_batch = _Const(batch_range.data(), {batch_size, 1, 1}, NCHW, halide_type_of<int>());
    VARP grid_x = _Const(height_range.data(), {1, height, 1}, NCHW, halide_type_of<int>());
    VARP grid_y = _Const(width_range.data(), {1, 1, width}, NCHW, halide_type_of<int>());

    grid_batch = _Tile(grid_batch, vectorToVARP({1, height, width}));
    grid_x = _Tile(grid_x, vectorToVARP({batch_size, 1, width}));
    grid_y = _Tile(grid_y, vectorToVARP({batch_size, height, 1}));

    grid_x = clamp(grid_x  + translation_x_var + _Scalar<int>(1), _Scalar<int>(0), _Scalar<int>(height + 1));
    grid_y = clamp(grid_y  + translation_y_var + _Scalar<int>(1), _Scalar<int>(0), _Scalar<int>(width + 1));
    VARP x_pad = _Pad(x, vectorToVARP({0, 0, 1, 1, 1, 1, 0, 0}));
    VARP x_perm = _Transpose(x_pad, {0, 2, 3, 1});
    VARP indices = _Stack({grid_batch, grid_x, grid_y}, 3);
    VARP selected = _GatherND(x_perm, indices);
    VARP result = _Transpose(selected, {0, 3, 1, 2});

    // auto result = x_pad->permute({0, 2, 3, 1}).index({grid[0], grid_x, grid_y}).permute({0, 3, 1, 2});
    return selected;
}

VARP rand_cutout(VARP x, float ratio_cutout) {
    // 获取输入张量的信息
    auto x_info = x->getInfo();
    int batch_size = x_info->dim[0];
    int height = x_info->dim[2];
    int width = x_info->dim[3];

    // 计算cutout_size
    int cutout_height = static_cast<int>(height * ratio_cutout + 0.5);
    int cutout_width = static_cast<int>(width * ratio_cutout + 0.5);

    // 生成随机的offset_x和offset_y
    std::vector<int> offset_x(batch_size, 0);
    std::vector<int> offset_y(batch_size, 0);
    for (int i = 0; i < batch_size; ++i) {
        // offset_x[i] = std::experimental::randint(0, height + (1 - cutout_height % 2));
        offset_x[i] = rand() % (height + (1 - cutout_height % 2) - 0 + 1) + 0;
        // offset_y[i] = std::experimental::randint(0, width + (1 - cutout_width % 2));
        offset_y[i] = rand() % (width + (1 - cutout_width % 2) - 0 + 1) + 0;

    }

    // 如果Siamese为真，设置所有offset_x和offset_y为相同的值
    // if (Siamese) {
    //     std::fill(offset_x.begin(), offset_x.end(), offset_x[0]);
    //     std::fill(offset_y.begin(), offset_y.end(), offset_y[0]);
    // }

    // 将offset_x和offset_y转换为MNN的VARP
    VARP offset_x_var = _Const(offset_x.data(), {batch_size, 1, 1}, NCHW, halide_type_of<int>());
    VARP offset_y_var = _Const(offset_y.data(), {batch_size, 1, 1}, NCHW, halide_type_of<int>());

    // 生成网格
    std::vector<int> batch_range(batch_size);
    std::vector<int> cutout_height_range(cutout_height);
    std::vector<int> cutout_width_range(cutout_width);
    std::iota(batch_range.begin(), batch_range.end(), 0);
    std::iota(cutout_height_range.begin(), cutout_height_range.end(), 0);
    std::iota(cutout_width_range.begin(), cutout_width_range.end(), 0);

    VARP grid_batch = _Const(batch_range.data(), {batch_size, 1, 1}, NCHW, halide_type_of<int>());
    VARP grid_x = _Const(cutout_height_range.data(), {1, cutout_height, 1}, NCHW, halide_type_of<int>());
    VARP grid_y = _Const(cutout_width_range.data(), {1, 1, cutout_width}, NCHW, halide_type_of<int>());

    grid_batch = _Tile(grid_batch, vectorToVARP({1, cutout_height, cutout_width}));
    grid_x = _Tile(grid_x, vectorToVARP({batch_size, 1, cutout_width}));
    grid_y = _Tile(grid_y, vectorToVARP({batch_size, cutout_height, 1}));

    // 调整grid_x和grid_y
    grid_x = clamp(grid_x + offset_x_var - _Scalar<int>(cutout_height) / _Scalar<int>(2), _Scalar<int>(0), _Scalar<int>(height - 1));
    grid_y = clamp(grid_y + offset_y_var - _Scalar<int>(cutout_width) / _Scalar<int>(2), _Scalar<int>(0), _Scalar<int>(width - 1));

    // 创建mask
    std::vector<float> ones(batch_size * height * width, 1.0f);
    VARP mask = _Const(ones.data(), {batch_size, height, width}, NCHW, halide_type_of<float>());
    mask = _ScatterNd(mask, _Stack({grid_batch, grid_x, grid_y}, 3), _ZerosLike(mask));
    mask = _Reshape(mask, {batch_size, 1, height, width});

    // 将mask应用到x上
    VARP result = x * mask;

    return result;
}

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
    cv::normalize(cv_clamp(de_img,0,1), norm_mat, 0, 255, cv::NORM_MINMAX);
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

    std::unordered_map<std::string, std::vector<std::function<VARP(VARP, float)>>> AUGMENT_FNS = {
        {"color", {rand_brightness, rand_saturation, rand_contrast}},
        {"crop", {rand_crop}},
        {"cutout", {rand_cutout}},
        {"flip", {rand_flip}},
        {"scale", {rand_scale}},
        {"rotate", {rand_rotate}},
    };

    std::string strategy = "color_crop_cutout_flip_scale_rotate";
    std::string delimiter = "_";

    size_t pos = 0;
    std::string token;
    while ((pos = strategy.find(delimiter)) != std::string::npos) {
        token = strategy.substr(0, pos);
        std::cout << token << std::endl;
        strategy.erase(0, pos + delimiter.length());
    }
    
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
    
    auto data = labels->readMap<uint8_t>();
    // // 根据标签将索引添加到 indices_class 中
    for (size_t i = 0; i < dataset_size; ++i) {
        uint8_t lab = data[i];
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
            syn_inputs.push_back(syn_input);
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
            syn_inputs.push_back(syn_input);
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
                        real_inputs.emplace_back(example.first[0]);
                        
                    }
                    auto real_feature = model->embedding(_Convert(_Concat(real_inputs,0), NCHW));
                    auto syn_feature = model->embedding(_Convert(_Concat(syn_inputs,0), NCHW));
                    loss = loss + _ReduceSum(_Square(_ReduceMean(_Reshape(real_feature, {numClasses,batchSize, -1}),{1}) - _ReduceMean(_Reshape(syn_feature, {numClasses,ipc, -1}),{1})));
                }else {
                    for (uint8_t c = 0; c < numClasses; c++){
                    
                        trainDataLoader->select_class(c);
                        auto example  = trainDataLoader->next(batchSize)[0];
                        auto syn_input = syn_inputs[c];
                        auto real_feature = model->embedding(_Convert(example.first[0], NCHW));
                        auto syn_feature = model->embedding(_Convert(syn_input, NCHW));
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

            }
            
            
        }

    }
}
