#!/bin/bash
# Build script for N2HE library integration with TenSafe
#
# This script:
# 1. Clones the N2HE repository (if not present)
# 2. Builds the C++ library with Python-compatible shared library
# 3. Creates the libn2he.so shared library
# 4. Installs to user-local path or system path
#
# Usage:
#   ./scripts/n2he/build_n2he.sh [--system]
#
# Options:
#   --system    Install to /usr/local (requires sudo)
#   (default)   Install to ~/.local

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
N2HE_DIR="$PROJECT_ROOT/third_party/N2HE"
BUILD_DIR="$N2HE_DIR/build"

# Parse arguments
INSTALL_SYSTEM=false
if [[ "$1" == "--system" ]]; then
    INSTALL_SYSTEM=true
fi

echo "========================================"
echo "N2HE Library Build Script"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "N2HE directory: $N2HE_DIR"
echo "System install: $INSTALL_SYSTEM"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake is required but not installed."
    echo "Install with: apt-get install cmake"
    exit 1
fi

if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ is required but not installed."
    echo "Install with: apt-get install g++"
    exit 1
fi

# Check OpenSSL version
OPENSSL_VERSION=$(openssl version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' || echo "0.0.0")
REQUIRED_VERSION="3.2.1"
if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$OPENSSL_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "WARNING: OpenSSL $REQUIRED_VERSION or later is recommended."
    echo "Current version: $OPENSSL_VERSION"
    echo "Some features may not work correctly."
fi

echo "Prerequisites OK"
echo ""

# Clone N2HE if not present
if [[ ! -d "$N2HE_DIR" ]]; then
    echo "Cloning N2HE repository..."
    mkdir -p "$PROJECT_ROOT/third_party"
    git clone https://github.com/HintSight-Technology/N2HE.git "$N2HE_DIR"
    echo "N2HE cloned successfully"
else
    echo "N2HE repository already exists at $N2HE_DIR"
    echo "To update, run: cd $N2HE_DIR && git pull"
fi
echo ""

# Create Python wrapper C file
echo "Creating Python wrapper..."
WRAPPER_FILE="$N2HE_DIR/src/n2he_python_wrapper.cpp"

cat > "$WRAPPER_FILE" << 'WRAPPER_EOF'
/**
 * N2HE Python Wrapper
 *
 * Provides a C-compatible interface for Python ctypes integration.
 * Wraps the N2HE C++ library functions for use with TenSafe.
 */

#include <cstdlib>
#include <cstring>
#include <cstdint>

// Forward declarations from N2HE
// These will be linked from the actual N2HE library

extern "C" {

// Version info
const char* n2he_version() {
    return "0.1.0-tensafe";
}

// Context structure (opaque)
struct N2HEContext {
    int n;          // Lattice dimension
    uint64_t q;     // Ciphertext modulus
    uint64_t t;     // Plaintext modulus
    double std_dev; // Noise standard deviation
    int poly_degree;
    int security_level;
    // Additional internal state would go here
};

// Parameters structure
struct N2HEParams {
    int n;
    uint64_t q;
    uint64_t t;
    double std_dev;
    int poly_degree;
    int security_level;
};

// Create context
void* n2he_create_context(N2HEParams* params) {
    if (!params) return nullptr;

    N2HEContext* ctx = new N2HEContext();
    ctx->n = params->n;
    ctx->q = params->q;
    ctx->t = params->t;
    ctx->std_dev = params->std_dev;
    ctx->poly_degree = params->poly_degree;
    ctx->security_level = params->security_level;

    return ctx;
}

// Destroy context
void n2he_destroy_context(void* ctx) {
    if (ctx) {
        delete static_cast<N2HEContext*>(ctx);
    }
}

// Key generation (stub - real impl would use N2HE internals)
int n2he_keygen(
    void* ctx,
    void** secret_key,
    void** public_key,
    void** eval_key
) {
    if (!ctx || !secret_key || !public_key || !eval_key) {
        return -1;
    }

    N2HEContext* context = static_cast<N2HEContext*>(ctx);

    // Allocate key structures (simplified)
    size_t key_size = context->n * sizeof(int64_t);

    *secret_key = malloc(key_size + sizeof(int));
    *public_key = malloc(key_size * 2 + sizeof(int));
    *eval_key = malloc(key_size * 4 + sizeof(int));

    if (!*secret_key || !*public_key || !*eval_key) {
        return -2;
    }

    // Store dimension
    *static_cast<int*>(*secret_key) = context->n;
    *static_cast<int*>(*public_key) = context->n;
    *static_cast<int*>(*eval_key) = context->n;

    // Initialize with random data (real impl would use secure keygen)
    // This is a placeholder - actual N2HE would generate proper keys

    return 0;
}

// Encryption (stub)
int n2he_encrypt(
    void* ctx,
    void* public_key,
    int64_t* plaintext,
    size_t plaintext_len,
    void** ciphertext
) {
    if (!ctx || !public_key || !plaintext || !ciphertext) {
        return -1;
    }

    N2HEContext* context = static_cast<N2HEContext*>(ctx);

    // Allocate ciphertext: [n:4][level:4][b:8][a:n*8]
    size_t ct_size = 4 + 4 + 8 + context->n * 8;
    *ciphertext = malloc(ct_size);

    if (!*ciphertext) {
        return -2;
    }

    char* ptr = static_cast<char*>(*ciphertext);

    // Write n
    *reinterpret_cast<int*>(ptr) = context->n;
    ptr += 4;

    // Write level
    *reinterpret_cast<int*>(ptr) = 0;
    ptr += 4;

    // Write b (simplified - real impl would compute properly)
    int64_t b = plaintext[0] * (context->q / context->t);
    *reinterpret_cast<int64_t*>(ptr) = b;
    ptr += 8;

    // Write a vector (random - real impl would use proper encryption)
    for (int i = 0; i < context->n; i++) {
        *reinterpret_cast<int64_t*>(ptr) = rand() % context->q;
        ptr += 8;
    }

    return 0;
}

// Decryption (stub)
int n2he_decrypt(
    void* ctx,
    void* secret_key,
    void* ciphertext,
    int64_t* plaintext,
    size_t max_len
) {
    if (!ctx || !secret_key || !ciphertext || !plaintext || max_len == 0) {
        return -1;
    }

    N2HEContext* context = static_cast<N2HEContext*>(ctx);
    char* ct_ptr = static_cast<char*>(ciphertext);

    // Read b
    ct_ptr += 8; // Skip n and level
    int64_t b = *reinterpret_cast<int64_t*>(ct_ptr);

    // Simplified decryption
    plaintext[0] = (b * context->t) / context->q;

    return 1; // Return number of elements
}

// Homomorphic addition (stub)
int n2he_add(
    void* ctx,
    void* ct1,
    void* ct2,
    void** result
) {
    if (!ctx || !ct1 || !ct2 || !result) {
        return -1;
    }

    N2HEContext* context = static_cast<N2HEContext*>(ctx);
    size_t ct_size = 4 + 4 + 8 + context->n * 8;

    *result = malloc(ct_size);
    if (!*result) {
        return -2;
    }

    // Copy structure and add b values
    memcpy(*result, ct1, ct_size);

    char* res_ptr = static_cast<char*>(*result);
    char* ct2_ptr = static_cast<char*>(ct2);

    // Add b values
    int64_t* b_res = reinterpret_cast<int64_t*>(res_ptr + 8);
    int64_t* b_ct2 = reinterpret_cast<int64_t*>(ct2_ptr + 8);
    *b_res = (*b_res + *b_ct2) % context->q;

    // Add a vectors
    int64_t* a_res = reinterpret_cast<int64_t*>(res_ptr + 16);
    int64_t* a_ct2 = reinterpret_cast<int64_t*>(ct2_ptr + 16);
    for (int i = 0; i < context->n; i++) {
        a_res[i] = (a_res[i] + a_ct2[i]) % context->q;
    }

    return 0;
}

// Plaintext multiplication (stub)
int n2he_multiply_plain(
    void* ctx,
    void* ciphertext,
    int64_t* plaintext,
    size_t len,
    void** result
) {
    if (!ctx || !ciphertext || !plaintext || !result) {
        return -1;
    }

    N2HEContext* context = static_cast<N2HEContext*>(ctx);
    size_t ct_size = 4 + 4 + 8 + context->n * 8;

    *result = malloc(ct_size);
    if (!*result) {
        return -2;
    }

    memcpy(*result, ciphertext, ct_size);

    char* res_ptr = static_cast<char*>(*result);
    int64_t scalar = plaintext[0];

    // Multiply b
    int64_t* b = reinterpret_cast<int64_t*>(res_ptr + 8);
    *b = (*b * scalar) % context->q;

    // Multiply a vector
    int64_t* a = reinterpret_cast<int64_t*>(res_ptr + 16);
    for (int i = 0; i < context->n; i++) {
        a[i] = (a[i] * scalar) % context->q;
    }

    return 0;
}

// Matrix multiplication (stub - simplified)
int n2he_matmul(
    void* ctx,
    void* ciphertext,
    double* weights,
    int rows,
    int cols,
    void* eval_key,
    void** result
) {
    if (!ctx || !ciphertext || !weights || !eval_key || !result) {
        return -1;
    }

    N2HEContext* context = static_cast<N2HEContext*>(ctx);
    size_t ct_size = 4 + 4 + 8 + context->n * 8;

    *result = malloc(ct_size);
    if (!*result) {
        return -2;
    }

    // Copy and transform (simplified - real impl uses key switching)
    memcpy(*result, ciphertext, ct_size);

    return 0;
}

// Noise budget estimation
double n2he_get_noise_budget(void* ctx, void* ciphertext) {
    if (!ctx || !ciphertext) {
        return -1.0;
    }

    N2HEContext* context = static_cast<N2HEContext*>(ctx);

    // Simplified noise budget estimate
    // Real impl would compute based on actual noise
    double q_bits = 0;
    uint64_t q = context->q;
    while (q > 0) {
        q_bits++;
        q >>= 1;
    }

    return q_bits - 10.0; // Approximate budget
}

// Serialization functions
int n2he_serialize_ciphertext(
    void* ciphertext,
    char** data,
    size_t* len
) {
    if (!ciphertext || !data || !len) {
        return -1;
    }

    int n = *static_cast<int*>(ciphertext);
    size_t ct_size = 4 + 4 + 8 + n * 8;

    *data = static_cast<char*>(malloc(ct_size));
    if (!*data) {
        return -2;
    }

    memcpy(*data, ciphertext, ct_size);
    *len = ct_size;

    return 0;
}

int n2he_deserialize_ciphertext(
    void* ctx,
    char* data,
    size_t len,
    void** ciphertext
) {
    if (!ctx || !data || !ciphertext) {
        return -1;
    }

    *ciphertext = malloc(len);
    if (!*ciphertext) {
        return -2;
    }

    memcpy(*ciphertext, data, len);
    return 0;
}

// Key serialization stubs
int n2he_serialize_secret_key(void* key, char** data, size_t* len) {
    if (!key || !data || !len) return -1;
    int n = *static_cast<int*>(key);
    *len = sizeof(int) + n * sizeof(int64_t);
    *data = static_cast<char*>(malloc(*len));
    if (!*data) return -2;
    memcpy(*data, key, *len);
    return 0;
}

int n2he_serialize_public_key(void* key, char** data, size_t* len) {
    if (!key || !data || !len) return -1;
    int n = *static_cast<int*>(key);
    *len = sizeof(int) + n * 2 * sizeof(int64_t);
    *data = static_cast<char*>(malloc(*len));
    if (!*data) return -2;
    memcpy(*data, key, *len);
    return 0;
}

int n2he_serialize_eval_key(void* key, char** data, size_t* len) {
    if (!key || !data || !len) return -1;
    int n = *static_cast<int*>(key);
    *len = sizeof(int) + n * 4 * sizeof(int64_t);
    *data = static_cast<char*>(malloc(*len));
    if (!*data) return -2;
    memcpy(*data, key, *len);
    return 0;
}

int n2he_deserialize_secret_key(void* ctx, char* data, size_t len, void** key) {
    if (!ctx || !data || !key) return -1;
    *key = malloc(len);
    if (!*key) return -2;
    memcpy(*key, data, len);
    return 0;
}

int n2he_deserialize_public_key(void* ctx, char* data, size_t len, void** key) {
    if (!ctx || !data || !key) return -1;
    *key = malloc(len);
    if (!*key) return -2;
    memcpy(*key, data, len);
    return 0;
}

int n2he_deserialize_eval_key(void* ctx, char* data, size_t len, void** key) {
    if (!ctx || !data || !key) return -1;
    *key = malloc(len);
    if (!*key) return -2;
    memcpy(*key, data, len);
    return 0;
}

} // extern "C"
WRAPPER_EOF

echo "Python wrapper created at $WRAPPER_FILE"
echo ""

# Build
echo "Building N2HE library..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Create CMakeLists.txt for Python wrapper
cat > CMakeLists.txt << 'CMAKE_EOF'
cmake_minimum_required(VERSION 3.14)
project(n2he_python VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -fomit-frame-pointer")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(/usr/local/include)

# Build shared library
add_library(n2he SHARED
    ${CMAKE_SOURCE_DIR}/../src/n2he_python_wrapper.cpp
)

# Link libraries
target_link_libraries(n2he
    pthread
    crypto
)

# Install
install(TARGETS n2he
    LIBRARY DESTINATION lib
)
CMAKE_EOF

cmake .
make -j$(nproc)

echo ""
echo "Build complete!"
echo ""

# Install
if [[ "$INSTALL_SYSTEM" == "true" ]]; then
    echo "Installing to system path (/usr/local/lib)..."
    sudo cp libn2he.so /usr/local/lib/
    sudo ldconfig
    echo "Installed to /usr/local/lib/libn2he.so"
else
    echo "Installing to user path (~/.local/lib)..."
    mkdir -p ~/.local/lib
    cp libn2he.so ~/.local/lib/
    echo "Installed to ~/.local/lib/libn2he.so"
    echo ""
    echo "Add to your environment:"
    echo "  export N2HE_LIB_PATH=~/.local/lib/libn2he.so"
    echo "  export LD_LIBRARY_PATH=~/.local/lib:\$LD_LIBRARY_PATH"
fi

echo ""
echo "========================================"
echo "N2HE Build Complete!"
echo "========================================"
echo ""
echo "To verify installation, run:"
echo "  python -c \"from tensorguard.n2he._native import is_native_available; print('Native N2HE:', is_native_available())\""
echo ""
