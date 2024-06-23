//
//  ClassDataLoader.hpp
//  MNN
//
//  Created by MNN on 2019/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ClassDataLoader_hpp
#define ClassDataLoader_hpp

#include <string>
#include <thread>
#include <vector>
#include "BlockingQueue.hpp"
#include "DataLoaderConfig.hpp"
#include "Example.hpp"
#include "ClassSampler.hpp"
namespace MNN {
namespace Train {
class BatchDataset;
class Sampler;
class ClassSampler;
class BatchTransform;
class MNN_PUBLIC ClassDataLoader {
public:
    ClassDataLoader(std::shared_ptr<BatchDataset> dataset, std::shared_ptr<ClassSampler> sampler,
               std::shared_ptr<DataLoaderConfig> config);
    /*
     When use Windows v141 toolset to compile class having vector of non-copyable element (std::thread, for example),
     copy constructor (or assignment operator) must be deleted explicity, otherwise compile will failed.
     */
    ClassDataLoader(const ClassDataLoader&) = delete;
    ClassDataLoader& operator = (const ClassDataLoader&) = delete;

    virtual ~ClassDataLoader() {
        join();
    };

    void prefetch(size_t nJobs);

    void workerThread();

    void join();

    std::vector<Example> next(size_t batchSize);

    void reset(uint8_t c);
    void select_class(uint8_t c);
    void clean(uint8_t c);

    size_t iterNumber() const;
    size_t size() const;
    static ClassDataLoader* makeDataLoader(std::shared_ptr<BatchDataset> dataset,
                                      const int batchSize,
                                      std::vector<std::vector<size_t>>& indices_class,
                                      const bool stack = true,
                                      const bool shuffle = true,
                                      const int numWorkers = 0);
    static ClassDataLoader* makeDataLoader(std::shared_ptr<BatchDataset> dataset,
                                      std::vector<std::shared_ptr<BatchTransform>> transforms,
                                      const int batchSize,
                                      std::vector<std::vector<size_t>>& indices_class,
                                      const bool shuffle = true,
                                      const int numWorkers = 0);

private:
    struct Job {
        std::vector<size_t> job;
        bool quit = false;
    };
    std::shared_ptr<BatchDataset> mDataset;
    std::shared_ptr<ClassSampler> mSampler;
    std::shared_ptr<DataLoaderConfig> mConfig;
    std::shared_ptr<BlockingQueue<Job>> mJobs;
    std::shared_ptr<BlockingQueue<std::vector<Example>>> mDataQueue;
    std::vector<std::thread> mWorkers;
};

} // namespace Train
} // namespace MNN

#endif // DataLoader_hpp
