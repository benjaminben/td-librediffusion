// Header-only meter for inference timing. Owns a small ring of cudaEvent
// pairs that bracket the inference call on a CUDA stream. record_start/end
// enqueue the markers; tick() polls completed pairs non-blockingly and
// updates an EMA of per-call GPU time.
//
// Reporting semantics (read by getInfoCHOPChan):
//   disabled        -> -1   (toggle off, metric is not being tracked)
//   enabled, idle   ->  0   (no inference fired within kStaleMs)
//   enabled, live   ->  EMA value

#pragma once

#include <cuda_runtime.h>

#include <chrono>

namespace librediff_td
{

class FpsMeter
{
public:
    static constexpr int kRingSize = 8;
    static constexpr double kStaleMs = 250.0;
    static constexpr double kEmaAlpha = 0.2;

    FpsMeter() = default;

    ~FpsMeter()
    {
        if(myCreated)
            destroyEvents();
    }

    FpsMeter(const FpsMeter&) = delete;
    FpsMeter& operator=(const FpsMeter&) = delete;

    void set_enabled(bool on) { myEnabled = on; }
    bool enabled() const { return myEnabled; }

    // Call before launching inference work on `stream`. Lazily creates
    // events on first use; safe to call from inside beginCUDAOperations.
    void record_start(cudaStream_t stream)
    {
        if(!myEnabled)
            return;
        if(!myCreated && !createEvents())
            return;
        cudaEventRecord(myStarts[myWriteIdx], stream);
        myInFlight[myWriteIdx] = true;
        myLastRecordTime = std::chrono::steady_clock::now();
    }

    // Call immediately after the inference call returns.
    void record_end(cudaStream_t stream)
    {
        if(!myEnabled || !myCreated)
            return;
        cudaEventRecord(myEnds[myWriteIdx], stream);
        myWriteIdx = (myWriteIdx + 1) % kRingSize;
    }

    // Drain any completed pairs and update the EMA. Non-blocking.
    void tick()
    {
        if(!myEnabled || !myCreated)
            return;
        for(int i = 0; i < kRingSize; ++i)
        {
            if(!myInFlight[i])
                continue;
            if(cudaEventQuery(myEnds[i]) != cudaSuccess)
                continue;
            float ms = 0.0f;
            if(cudaEventElapsedTime(&ms, myStarts[i], myEnds[i]) == cudaSuccess)
            {
                if(myEmaMs <= 0.0)
                    myEmaMs = ms;
                else
                    myEmaMs = kEmaAlpha * ms + (1.0 - kEmaAlpha) * myEmaMs;
            }
            myInFlight[i] = false;
        }
    }

    double inference_ms() const
    {
        if(!myEnabled)
            return -1.0;
        if(isStale() || myEmaMs <= 0.0)
            return 0.0;
        return myEmaMs;
    }

    double inference_fps() const
    {
        if(!myEnabled)
            return -1.0;
        if(isStale() || myEmaMs <= 0.0)
            return 0.0;
        return 1000.0 / myEmaMs;
    }

private:
    bool isStale() const
    {
        if(myLastRecordTime.time_since_epoch().count() == 0)
            return true;
        auto now = std::chrono::steady_clock::now();
        double delta = std::chrono::duration<double, std::milli>(
                           now - myLastRecordTime).count();
        return delta > kStaleMs;
    }

    bool createEvents()
    {
        for(int i = 0; i < kRingSize; ++i)
        {
            if(cudaEventCreate(&myStarts[i]) != cudaSuccess
               || cudaEventCreate(&myEnds[i]) != cudaSuccess)
            {
                destroyEvents();
                return false;
            }
            myInFlight[i] = false;
        }
        myCreated = true;
        myWriteIdx = 0;
        myEmaMs = 0.0;
        return true;
    }

    void destroyEvents()
    {
        for(int i = 0; i < kRingSize; ++i)
        {
            if(myStarts[i])
                cudaEventDestroy(myStarts[i]);
            if(myEnds[i])
                cudaEventDestroy(myEnds[i]);
            myStarts[i] = nullptr;
            myEnds[i] = nullptr;
            myInFlight[i] = false;
        }
        myCreated = false;
    }

    bool myEnabled = false;
    bool myCreated = false;
    int myWriteIdx = 0;
    double myEmaMs = 0.0;
    cudaEvent_t myStarts[kRingSize] = {};
    cudaEvent_t myEnds[kRingSize] = {};
    bool myInFlight[kRingSize] = {};
    std::chrono::steady_clock::time_point myLastRecordTime{};
};

} // namespace librediff_td
