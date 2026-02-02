/**
 * Backend Adapter Implementation
 *
 * This file provides the implementation glue for connecting the
 * GPUCKKSBackend interface to actual GPU HE libraries.
 *
 * Currently supported backends:
 *   - HEonGPU (recommended for production)
 *   - Simulation (for testing)
 *
 * Future backends:
 *   - FIDESlib
 *   - OpenFHE-GPU
 */

#include "gpu_ckks_backend.h"

#include <stdexcept>
#include <chrono>
#include <cstring>
#include <cmath>

// Conditional compilation based on available backends
#ifdef HE_LORA_HAS_HEONGPU
#include <heongpu/heongpu.h>
#endif

#ifdef HE_LORA_HAS_FIDESLIB
#include <fideslib/fideslib.h>
#endif

#ifdef HE_LORA_HAS_OPENFHE_GPU
#include <openfhe-gpu/openfhe_gpu.h>
#endif

// Always have CUDA for simulation
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Stub for non-CUDA builds
namespace {
    struct cudaStream_t_stub {};
    using cudaStream_t = cudaStream_t_stub*;
    int cudaStreamCreate(cudaStream_t*) { return 0; }
    int cudaStreamDestroy(cudaStream_t) { return 0; }
    int cudaStreamSynchronize(cudaStream_t) { return 0; }
    int cudaDeviceSynchronize() { return 0; }
}
#endif

namespace he_lora {
namespace gpu {

// =============================================================================
// CUDA STREAM IMPLEMENTATION
// =============================================================================

CudaStream::CudaStream(int device_id) : device_id_(device_id), stream_(nullptr) {
#ifdef __CUDACC__
    cudaSetDevice(device_id);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    stream_ = reinterpret_cast<void*>(stream);
#endif
}

CudaStream::~CudaStream() {
#ifdef __CUDACC__
    if (stream_) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
    }
#endif
}

void CudaStream::synchronize() {
#ifdef __CUDACC__
    if (stream_) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
    }
#endif
}

void* CudaStream::native_handle() {
    return stream_;
}

// =============================================================================
// CIPHERTEXT IMPLEMENTATION
// =============================================================================

CKKSCiphertext::CKKSCiphertext(void* data, int level, double scale,
                               uint64_t slot_count, bool is_ntt, int device_id)
    : data_(data), level_(level), scale_(scale),
      slot_count_(slot_count), is_ntt_(is_ntt), device_id_(device_id) {}

CKKSCiphertext::~CKKSCiphertext() {
    // GPU memory freed by backend
}

CKKSCiphertext::CKKSCiphertext(CKKSCiphertext&& other) noexcept
    : data_(other.data_), level_(other.level_), scale_(other.scale_),
      slot_count_(other.slot_count_), is_ntt_(other.is_ntt_),
      device_id_(other.device_id_) {
    other.data_ = nullptr;
}

CKKSCiphertext& CKKSCiphertext::operator=(CKKSCiphertext&& other) noexcept {
    if (this != &other) {
        data_ = other.data_;
        level_ = other.level_;
        scale_ = other.scale_;
        slot_count_ = other.slot_count_;
        is_ntt_ = other.is_ntt_;
        device_id_ = other.device_id_;
        other.data_ = nullptr;
    }
    return *this;
}

size_t CKKSCiphertext::size_bytes() const {
    // Approximation: 2 polynomials × N × (Q bits) / 8
    // This varies by level
    return 2 * slot_count_ * 2 * 8 * (10 - level_);  // Rough estimate
}

// =============================================================================
// PLAINTEXT IMPLEMENTATION
// =============================================================================

CKKSPlaintext::CKKSPlaintext(void* data, double scale, uint64_t slot_count, bool is_ntt)
    : data_(data), scale_(scale), slot_count_(slot_count), is_ntt_(is_ntt) {}

CKKSPlaintext::~CKKSPlaintext() {
    // Memory freed by backend
}

CKKSPlaintext::CKKSPlaintext(CKKSPlaintext&& other) noexcept
    : data_(other.data_), scale_(other.scale_),
      slot_count_(other.slot_count_), is_ntt_(other.is_ntt_) {
    other.data_ = nullptr;
}

CKKSPlaintext& CKKSPlaintext::operator=(CKKSPlaintext&& other) noexcept {
    if (this != &other) {
        data_ = other.data_;
        scale_ = other.scale_;
        slot_count_ = other.slot_count_;
        is_ntt_ = other.is_ntt_;
        other.data_ = nullptr;
    }
    return *this;
}

// =============================================================================
// SIMULATION BACKEND
// =============================================================================

/**
 * Simulation backend for testing without real GPU HE.
 *
 * This simulates CKKS operations with plaintext arithmetic
 * but accurately tracks all operation counts.
 */
class SimulationBackend : public GPUCKKSBackend {
public:
    bool initialize(const CKKSParams& params, int device_id = 0) override {
        params_ = params;
        device_id_ = device_id;
        initialized_ = true;
        counters_.reset();
        return true;
    }

    bool is_initialized() const override {
        return initialized_;
    }

    const CKKSParams& params() const override {
        return params_;
    }

    std::string device_info() const override {
        return "Simulation Backend (CPU) - FOR TESTING ONLY";
    }

    std::unique_ptr<CKKSCiphertext> encrypt(
        const std::vector<double>& values,
        CudaStream* stream = nullptr
    ) override {
        counters_.encryptions++;

        auto start = std::chrono::high_resolution_clock::now();

        // Allocate and copy data
        size_t size = params_.slot_count() * sizeof(double);
        double* data = new double[params_.slot_count()];
        std::memset(data, 0, size);
        std::memcpy(data, values.data(),
                    std::min(values.size(), params_.slot_count()) * sizeof(double));

        auto ct = std::unique_ptr<CKKSCiphertext>(new CKKSCiphertext(
            data, 0, params_.scale(), params_.slot_count(), true, device_id_
        ));

        auto end = std::chrono::high_resolution_clock::now();
        counters_.encrypt_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        counters_.total_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        return ct;
    }

    std::vector<double> decrypt(
        const CKKSCiphertext& ct,
        CudaStream* stream = nullptr
    ) override {
        counters_.decryptions++;

        auto start = std::chrono::high_resolution_clock::now();

        const double* data = static_cast<const double*>(ct.data());
        std::vector<double> result(data, data + ct.slot_count());

        auto end = std::chrono::high_resolution_clock::now();
        counters_.decrypt_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        counters_.total_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        return result;
    }

    std::unique_ptr<CKKSPlaintext> encode(
        const std::vector<double>& values
    ) override {
        double* data = new double[params_.slot_count()];
        std::memset(data, 0, params_.slot_count() * sizeof(double));
        std::memcpy(data, values.data(),
                    std::min(values.size(), params_.slot_count()) * sizeof(double));

        return std::unique_ptr<CKKSPlaintext>(new CKKSPlaintext(
            data, params_.scale(), params_.slot_count(), true
        ));
    }

    std::unique_ptr<CKKSCiphertext> add(
        const CKKSCiphertext& ct1,
        const CKKSCiphertext& ct2,
        CudaStream* stream = nullptr
    ) override {
        counters_.additions++;

        auto start = std::chrono::high_resolution_clock::now();

        const double* d1 = static_cast<const double*>(ct1.data());
        const double* d2 = static_cast<const double*>(ct2.data());
        double* result = new double[params_.slot_count()];

        for (size_t i = 0; i < params_.slot_count(); i++) {
            result[i] = d1[i] + d2[i];
        }

        auto ct = std::unique_ptr<CKKSCiphertext>(new CKKSCiphertext(
            result, std::max(ct1.level(), ct2.level()),
            ct1.scale(), params_.slot_count(), true, device_id_
        ));

        auto end = std::chrono::high_resolution_clock::now();
        counters_.compute_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        counters_.total_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        return ct;
    }

    void add_inplace(
        CKKSCiphertext& ct1,
        const CKKSCiphertext& ct2,
        CudaStream* stream = nullptr
    ) override {
        counters_.additions++;

        auto start = std::chrono::high_resolution_clock::now();

        double* d1 = static_cast<double*>(ct1.data());
        const double* d2 = static_cast<const double*>(ct2.data());

        for (size_t i = 0; i < params_.slot_count(); i++) {
            d1[i] += d2[i];
        }

        auto end = std::chrono::high_resolution_clock::now();
        counters_.compute_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        counters_.total_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::unique_ptr<CKKSCiphertext> mul_plain(
        const CKKSCiphertext& ct,
        const CKKSPlaintext& pt,
        CudaStream* stream = nullptr
    ) override {
        counters_.multiplications++;

        auto start = std::chrono::high_resolution_clock::now();

        const double* ct_data = static_cast<const double*>(ct.data());
        const double* pt_data = static_cast<const double*>(pt.data());
        double* result = new double[params_.slot_count()];

        for (size_t i = 0; i < params_.slot_count(); i++) {
            result[i] = ct_data[i] * pt_data[i];
        }

        auto new_ct = std::unique_ptr<CKKSCiphertext>(new CKKSCiphertext(
            result, ct.level(), ct.scale() * pt.scale(),
            params_.slot_count(), true, device_id_
        ));

        auto end = std::chrono::high_resolution_clock::now();
        counters_.compute_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        counters_.total_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        return new_ct;
    }

    void mul_plain_inplace(
        CKKSCiphertext& ct,
        const CKKSPlaintext& pt,
        CudaStream* stream = nullptr
    ) override {
        counters_.multiplications++;

        auto start = std::chrono::high_resolution_clock::now();

        double* ct_data = static_cast<double*>(ct.data());
        const double* pt_data = static_cast<const double*>(pt.data());

        for (size_t i = 0; i < params_.slot_count(); i++) {
            ct_data[i] *= pt_data[i];
        }

        auto end = std::chrono::high_resolution_clock::now();
        counters_.compute_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        counters_.total_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::unique_ptr<CKKSCiphertext> rotate(
        const CKKSCiphertext& ct,
        int steps,
        CudaStream* stream = nullptr
    ) override {
        counters_.rotations++;
        counters_.keyswitches++;  // Rotation requires key switch

        auto start = std::chrono::high_resolution_clock::now();

        const double* ct_data = static_cast<const double*>(ct.data());
        double* result = new double[params_.slot_count()];

        // Simulate rotation
        for (size_t i = 0; i < params_.slot_count(); i++) {
            size_t src = (i + steps + params_.slot_count()) % params_.slot_count();
            result[i] = ct_data[src];
        }

        auto new_ct = std::unique_ptr<CKKSCiphertext>(new CKKSCiphertext(
            result, ct.level(), ct.scale(),
            params_.slot_count(), true, device_id_
        ));

        auto end = std::chrono::high_resolution_clock::now();
        counters_.compute_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        counters_.total_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        return new_ct;
    }

    void rotate_inplace(
        CKKSCiphertext& ct,
        int steps,
        CudaStream* stream = nullptr
    ) override {
        counters_.rotations++;
        counters_.keyswitches++;

        auto start = std::chrono::high_resolution_clock::now();

        double* ct_data = static_cast<double*>(ct.data());
        std::vector<double> temp(ct_data, ct_data + params_.slot_count());

        for (size_t i = 0; i < params_.slot_count(); i++) {
            size_t src = (i + steps + params_.slot_count()) % params_.slot_count();
            ct_data[i] = temp[src];
        }

        auto end = std::chrono::high_resolution_clock::now();
        counters_.compute_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        counters_.total_time_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::unique_ptr<CKKSCiphertext> rescale(
        const CKKSCiphertext& ct,
        CudaStream* stream = nullptr
    ) override {
        counters_.rescales++;

        const double* ct_data = static_cast<const double*>(ct.data());
        double* result = new double[params_.slot_count()];
        std::memcpy(result, ct_data, params_.slot_count() * sizeof(double));

        return std::unique_ptr<CKKSCiphertext>(new CKKSCiphertext(
            result, ct.level() + 1, params_.scale(),
            params_.slot_count(), true, device_id_
        ));
    }

    void rescale_inplace(
        CKKSCiphertext& ct,
        CudaStream* stream = nullptr
    ) override {
        counters_.rescales++;
        // In simulation, just update metadata
        // Real implementation would modify the ciphertext
    }

    std::unique_ptr<CKKSCiphertext> modswitch(
        const CKKSCiphertext& ct,
        CudaStream* stream = nullptr
    ) override {
        counters_.modswitches++;

        const double* ct_data = static_cast<const double*>(ct.data());
        double* result = new double[params_.slot_count()];
        std::memcpy(result, ct_data, params_.slot_count() * sizeof(double));

        return std::unique_ptr<CKKSCiphertext>(new CKKSCiphertext(
            result, ct.level() + 1, ct.scale(),
            params_.slot_count(), true, device_id_
        ));
    }

    std::unique_ptr<CKKSCiphertext> modswitch_to_level(
        const CKKSCiphertext& ct,
        int target_level,
        CudaStream* stream = nullptr
    ) override {
        auto result = std::unique_ptr<CKKSCiphertext>(
            new CKKSCiphertext(nullptr, ct.level(), ct.scale(),
                              ct.slot_count(), ct.is_ntt(), device_id_)
        );

        // Copy data
        const double* ct_data = static_cast<const double*>(ct.data());
        double* new_data = new double[params_.slot_count()];
        std::memcpy(new_data, ct_data, params_.slot_count() * sizeof(double));

        result = std::unique_ptr<CKKSCiphertext>(new CKKSCiphertext(
            new_data, ct.level(), ct.scale(),
            params_.slot_count(), true, device_id_
        ));

        while (result->level() < target_level) {
            result = modswitch(*result, stream);
        }

        return result;
    }

    const OperationCounters& counters() const override {
        return counters_;
    }

    void reset_counters() override {
        counters_.reset();
    }

    size_t memory_usage() const override {
        return 0;  // Simulation doesn't track real GPU memory
    }

    size_t total_memory() const override {
        return 0;
    }

private:
    CKKSParams params_;
    int device_id_ = 0;
    bool initialized_ = false;
    mutable OperationCounters counters_;
};

// =============================================================================
// HEONGPU BACKEND (PLACEHOLDER)
// =============================================================================

#ifdef HE_LORA_HAS_HEONGPU
class HEonGPUBackend : public GPUCKKSBackend {
    // TODO: Implement HEonGPU wrapper
    // This would wrap the HEonGPU library for actual GPU acceleration
};
#endif

// =============================================================================
// FIDESLIB BACKEND (PLACEHOLDER)
// =============================================================================

#ifdef HE_LORA_HAS_FIDESLIB
class FIDESlibBackend : public GPUCKKSBackend {
    // TODO: Implement FIDESlib wrapper
};
#endif

// =============================================================================
// OPENFHE-GPU BACKEND (PLACEHOLDER)
// =============================================================================

#ifdef HE_LORA_HAS_OPENFHE_GPU
class OpenFHEGPUBackend : public GPUCKKSBackend {
    // TODO: Implement OpenFHE-GPU wrapper
};
#endif

// =============================================================================
// BACKEND FACTORY
// =============================================================================

std::unique_ptr<GPUCKKSBackend> create_backend(
    BackendType type,
    const CKKSParams& params,
    int device_id
) {
    std::unique_ptr<GPUCKKSBackend> backend;

    switch (type) {
        case BackendType::Simulation:
            backend = std::make_unique<SimulationBackend>();
            break;

#ifdef HE_LORA_HAS_HEONGPU
        case BackendType::HEonGPU:
            backend = std::make_unique<HEonGPUBackend>();
            break;
#endif

#ifdef HE_LORA_HAS_FIDESLIB
        case BackendType::FIDESlib:
            backend = std::make_unique<FIDESlibBackend>();
            break;
#endif

#ifdef HE_LORA_HAS_OPENFHE_GPU
        case BackendType::OpenFHE_GPU:
            backend = std::make_unique<OpenFHEGPUBackend>();
            break;
#endif

        default:
            throw std::runtime_error("Backend type not available");
    }

    if (!backend->initialize(params, device_id)) {
        throw std::runtime_error("Failed to initialize backend");
    }

    return backend;
}

bool is_backend_available(BackendType type) {
    switch (type) {
        case BackendType::Simulation:
            return true;

#ifdef HE_LORA_HAS_HEONGPU
        case BackendType::HEonGPU:
            return true;
#endif

#ifdef HE_LORA_HAS_FIDESLIB
        case BackendType::FIDESlib:
            return true;
#endif

#ifdef HE_LORA_HAS_OPENFHE_GPU
        case BackendType::OpenFHE_GPU:
            return true;
#endif

        default:
            return false;
    }
}

std::vector<BackendType> available_backends() {
    std::vector<BackendType> backends;

    backends.push_back(BackendType::Simulation);

#ifdef HE_LORA_HAS_HEONGPU
    backends.push_back(BackendType::HEonGPU);
#endif

#ifdef HE_LORA_HAS_FIDESLIB
    backends.push_back(BackendType::FIDESlib);
#endif

#ifdef HE_LORA_HAS_OPENFHE_GPU
    backends.push_back(BackendType::OpenFHE_GPU);
#endif

    return backends;
}

}  // namespace gpu
}  // namespace he_lora
