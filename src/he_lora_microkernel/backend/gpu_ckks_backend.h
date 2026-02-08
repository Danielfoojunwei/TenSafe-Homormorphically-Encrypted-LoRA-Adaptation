/**
 * GPU CKKS Backend C++ Interface
 *
 * This header defines the C++ interface for GPU-accelerated CKKS backends.
 * It provides a common abstraction layer over different GPU HE libraries:
 *   - HEonGPU
 *   - FIDESlib
 *   - OpenFHE-GPU
 *
 * The interface is designed for:
 *   - GPU-resident ciphertexts and keys
 *   - Asynchronous execution with CUDA streams
 *   - Minimal memory transfers
 *   - Operation counting for cost tracking
 *
 * NO CPU FALLBACK - GPU execution is REQUIRED.
 */

#ifndef HE_LORA_MICROKERNEL_GPU_CKKS_BACKEND_H
#define HE_LORA_MICROKERNEL_GPU_CKKS_BACKEND_H

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace he_lora {
namespace gpu {

// =============================================================================
// FORWARD DECLARATIONS
// =============================================================================

class CKKSContext;
class CKKSCiphertext;
class CKKSPlaintext;
class CKKSKeys;

// =============================================================================
// CKKS PARAMETERS
// =============================================================================

/**
 * CKKS encryption parameters.
 */
struct CKKSParams {
    uint64_t poly_modulus_degree;      // N (ring dimension)
    std::vector<int> coeff_modulus_bits;  // Bit lengths of moduli
    int scale_bits;                     // Scale as power of 2

    // Derived
    uint64_t slot_count() const { return poly_modulus_degree / 2; }
    int max_depth() const { return static_cast<int>(coeff_modulus_bits.size()) - 2; }
    double scale() const { return static_cast<double>(1ULL << scale_bits); }
};

// =============================================================================
// OPERATION COUNTERS
// =============================================================================

/**
 * Counters for tracking HE operation costs.
 * Critical for rotation-minimal design verification.
 */
struct OperationCounters {
    uint64_t rotations = 0;
    uint64_t keyswitches = 0;
    uint64_t rescales = 0;
    uint64_t modswitches = 0;
    uint64_t multiplications = 0;
    uint64_t additions = 0;
    uint64_t encryptions = 0;
    uint64_t decryptions = 0;

    // Timing (microseconds)
    uint64_t total_time_us = 0;
    uint64_t encrypt_time_us = 0;
    uint64_t compute_time_us = 0;
    uint64_t decrypt_time_us = 0;

    void reset() {
        rotations = keyswitches = rescales = modswitches = 0;
        multiplications = additions = encryptions = decryptions = 0;
        total_time_us = encrypt_time_us = compute_time_us = decrypt_time_us = 0;
    }

    OperationCounters operator+(const OperationCounters& other) const {
        OperationCounters result;
        result.rotations = rotations + other.rotations;
        result.keyswitches = keyswitches + other.keyswitches;
        result.rescales = rescales + other.rescales;
        result.modswitches = modswitches + other.modswitches;
        result.multiplications = multiplications + other.multiplications;
        result.additions = additions + other.additions;
        result.encryptions = encryptions + other.encryptions;
        result.decryptions = decryptions + other.decryptions;
        result.total_time_us = total_time_us + other.total_time_us;
        result.encrypt_time_us = encrypt_time_us + other.encrypt_time_us;
        result.compute_time_us = compute_time_us + other.compute_time_us;
        result.decrypt_time_us = decrypt_time_us + other.decrypt_time_us;
        return result;
    }
};

// =============================================================================
// CUDA STREAM HANDLE
// =============================================================================

/**
 * Wrapper for CUDA stream management.
 */
class CudaStream {
public:
    explicit CudaStream(int device_id = 0);
    ~CudaStream();

    void synchronize();
    void* native_handle();  // Returns cudaStream_t
    int device_id() const { return device_id_; }

private:
    void* stream_;
    int device_id_;
};

// =============================================================================
// GPU CIPHERTEXT
// =============================================================================

/**
 * GPU-resident ciphertext.
 *
 * The actual data is stored on GPU memory.
 * This class manages the GPU allocation and provides metadata.
 */
class CKKSCiphertext {
public:
    // Metadata accessors
    int level() const { return level_; }
    double scale() const { return scale_; }
    uint64_t slot_count() const { return slot_count_; }
    bool is_ntt() const { return is_ntt_; }
    int device_id() const { return device_id_; }

    // Size information
    size_t size_bytes() const;

    // GPU data pointer (for backend implementation)
    void* data() { return data_; }
    const void* data() const { return data_; }

    // Move semantics (no copy - GPU resources)
    CKKSCiphertext(CKKSCiphertext&& other) noexcept;
    CKKSCiphertext& operator=(CKKSCiphertext&& other) noexcept;
    CKKSCiphertext(const CKKSCiphertext&) = delete;
    CKKSCiphertext& operator=(const CKKSCiphertext&) = delete;

    ~CKKSCiphertext();

private:
    friend class GPUCKKSBackend;

    CKKSCiphertext(void* data, int level, double scale,
                   uint64_t slot_count, bool is_ntt, int device_id);

    void* data_;
    int level_;
    double scale_;
    uint64_t slot_count_;
    bool is_ntt_;
    int device_id_;
};

// =============================================================================
// GPU PLAINTEXT (PRE-ENCODED)
// =============================================================================

/**
 * GPU-resident pre-encoded plaintext.
 *
 * Used for Ct×Pt multiplication where the plaintext represents
 * LoRA weight matrix blocks. These are encoded ONCE at compile time.
 */
class CKKSPlaintext {
public:
    double scale() const { return scale_; }
    uint64_t slot_count() const { return slot_count_; }
    bool is_ntt() const { return is_ntt_; }

    void* data() { return data_; }
    const void* data() const { return data_; }

    CKKSPlaintext(CKKSPlaintext&& other) noexcept;
    CKKSPlaintext& operator=(CKKSPlaintext&& other) noexcept;
    CKKSPlaintext(const CKKSPlaintext&) = delete;
    CKKSPlaintext& operator=(const CKKSPlaintext&) = delete;

    ~CKKSPlaintext();

private:
    friend class GPUCKKSBackend;

    CKKSPlaintext(void* data, double scale, uint64_t slot_count, bool is_ntt);

    void* data_;
    double scale_;
    uint64_t slot_count_;
    bool is_ntt_;
};

// =============================================================================
// GPU CKKS BACKEND INTERFACE
// =============================================================================

/**
 * Abstract interface for GPU CKKS backends.
 *
 * This is the main interface that backends must implement.
 * All operations are performed on GPU, and ciphertexts remain GPU-resident.
 */
class GPUCKKSBackend {
public:
    virtual ~GPUCKKSBackend() = default;

    // -------------------------------------------------------------------------
    // INITIALIZATION
    // -------------------------------------------------------------------------

    /**
     * Initialize the backend with generated keys.
     *
     * @param params CKKS parameters
     * @param device_id GPU device ID
     * @return true on success
     */
    virtual bool initialize(const CKKSParams& params, int device_id = 0) = 0;

    /**
     * Check if backend is initialized.
     */
    virtual bool is_initialized() const = 0;

    /**
     * Get CKKS parameters.
     */
    virtual const CKKSParams& params() const = 0;

    /**
     * Get device information string.
     */
    virtual std::string device_info() const = 0;

    // -------------------------------------------------------------------------
    // ENCRYPTION / DECRYPTION
    // -------------------------------------------------------------------------

    /**
     * Encrypt a vector of doubles.
     *
     * @param values Input vector (length <= slot_count)
     * @param stream CUDA stream for async execution
     * @return GPU-resident ciphertext
     */
    virtual std::unique_ptr<CKKSCiphertext> encrypt(
        const std::vector<double>& values,
        CudaStream* stream = nullptr
    ) = 0;

    /**
     * Decrypt a ciphertext to vector.
     *
     * @param ct Ciphertext to decrypt
     * @param stream CUDA stream for async execution
     * @return Decrypted values
     */
    virtual std::vector<double> decrypt(
        const CKKSCiphertext& ct,
        CudaStream* stream = nullptr
    ) = 0;

    /**
     * Encode values into plaintext (no encryption).
     *
     * @param values Input vector
     * @return GPU-resident encoded plaintext
     */
    virtual std::unique_ptr<CKKSPlaintext> encode(
        const std::vector<double>& values
    ) = 0;

    // -------------------------------------------------------------------------
    // ARITHMETIC
    // -------------------------------------------------------------------------

    /**
     * Add two ciphertexts.
     */
    virtual std::unique_ptr<CKKSCiphertext> add(
        const CKKSCiphertext& ct1,
        const CKKSCiphertext& ct2,
        CudaStream* stream = nullptr
    ) = 0;

    /**
     * Add ciphertexts in-place: ct1 += ct2
     */
    virtual void add_inplace(
        CKKSCiphertext& ct1,
        const CKKSCiphertext& ct2,
        CudaStream* stream = nullptr
    ) = 0;

    /**
     * Multiply ciphertext by plaintext (Ct×Pt).
     * This is the core LoRA operation.
     */
    virtual std::unique_ptr<CKKSCiphertext> mul_plain(
        const CKKSCiphertext& ct,
        const CKKSPlaintext& pt,
        CudaStream* stream = nullptr
    ) = 0;

    /**
     * Multiply in-place: ct *= pt
     */
    virtual void mul_plain_inplace(
        CKKSCiphertext& ct,
        const CKKSPlaintext& pt,
        CudaStream* stream = nullptr
    ) = 0;

    // -------------------------------------------------------------------------
    // ROTATION (CRITICAL FOR MOAI)
    // -------------------------------------------------------------------------

    /**
     * Rotate ciphertext slots.
     *
     * This is the MOST EXPENSIVE operation.
     * MOAI-style design minimizes rotations.
     *
     * @param ct Ciphertext to rotate
     * @param steps Rotation amount (positive = left)
     * @param stream CUDA stream
     * @return Rotated ciphertext
     */
    virtual std::unique_ptr<CKKSCiphertext> rotate(
        const CKKSCiphertext& ct,
        int steps,
        CudaStream* stream = nullptr
    ) = 0;

    /**
     * Rotate in-place.
     */
    virtual void rotate_inplace(
        CKKSCiphertext& ct,
        int steps,
        CudaStream* stream = nullptr
    ) = 0;

    // -------------------------------------------------------------------------
    // LEVEL MANAGEMENT
    // -------------------------------------------------------------------------

    /**
     * Rescale ciphertext after multiplication.
     */
    virtual std::unique_ptr<CKKSCiphertext> rescale(
        const CKKSCiphertext& ct,
        CudaStream* stream = nullptr
    ) = 0;

    /**
     * Rescale in-place.
     */
    virtual void rescale_inplace(
        CKKSCiphertext& ct,
        CudaStream* stream = nullptr
    ) = 0;

    /**
     * Modulus switch (level drop without rescaling).
     */
    virtual std::unique_ptr<CKKSCiphertext> modswitch(
        const CKKSCiphertext& ct,
        CudaStream* stream = nullptr
    ) = 0;

    /**
     * Modswitch to specific level.
     */
    virtual std::unique_ptr<CKKSCiphertext> modswitch_to_level(
        const CKKSCiphertext& ct,
        int target_level,
        CudaStream* stream = nullptr
    ) = 0;

    // -------------------------------------------------------------------------
    // FUSED OPERATIONS (KERNEL FUSION)
    // -------------------------------------------------------------------------

    /**
     * Fused multiply-rescale: rescale(ct × pt)
     * Default: call mul_plain then rescale.
     * Backends may override with fused GPU kernel.
     */
    virtual std::unique_ptr<CKKSCiphertext> mul_plain_rescale(
        const CKKSCiphertext& ct,
        const CKKSPlaintext& pt,
        CudaStream* stream = nullptr
    ) {
        auto result = mul_plain(ct, pt, stream);
        rescale_inplace(*result, stream);
        return result;
    }

    /**
     * Fused multiply-rescale-add: acc += rescale(ct × pt)
     */
    virtual void mul_plain_rescale_add(
        const CKKSCiphertext& ct,
        const CKKSPlaintext& pt,
        CKKSCiphertext& accumulator,
        CudaStream* stream = nullptr
    ) {
        auto product = mul_plain_rescale(ct, pt, stream);
        if (product->level() != accumulator.level()) {
            product = modswitch_to_level(*product, accumulator.level(), stream);
        }
        add_inplace(accumulator, *product, stream);
    }

    // -------------------------------------------------------------------------
    // COUNTERS
    // -------------------------------------------------------------------------

    /**
     * Get operation counters.
     */
    virtual const OperationCounters& counters() const = 0;

    /**
     * Reset operation counters.
     */
    virtual void reset_counters() = 0;

    // -------------------------------------------------------------------------
    // MEMORY MANAGEMENT
    // -------------------------------------------------------------------------

    /**
     * Get GPU memory usage in bytes.
     */
    virtual size_t memory_usage() const = 0;

    /**
     * Get total GPU memory in bytes.
     */
    virtual size_t total_memory() const = 0;
};

// =============================================================================
// BACKEND FACTORY
// =============================================================================

/**
 * Backend types supported.
 */
enum class BackendType {
    HEonGPU,
    FIDESlib,
    OpenFHE_GPU,
    Simulation  // For testing
};

/**
 * Create a GPU CKKS backend.
 *
 * @param type Backend implementation to use
 * @param params CKKS parameters
 * @param device_id GPU device ID
 * @return Initialized backend (throws on failure)
 */
std::unique_ptr<GPUCKKSBackend> create_backend(
    BackendType type,
    const CKKSParams& params,
    int device_id = 0
);

/**
 * Check if a backend type is available.
 */
bool is_backend_available(BackendType type);

/**
 * Get list of available backends.
 */
std::vector<BackendType> available_backends();

}  // namespace gpu
}  // namespace he_lora

#endif  // HE_LORA_MICROKERNEL_GPU_CKKS_BACKEND_H
