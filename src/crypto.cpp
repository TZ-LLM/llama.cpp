#include "crypto.h"
#include "interface.h"
#include <cstring>
#include <optional>
#include <atomic>
#include "ggml.h"

#include <openssl/evp.h>
#include <openssl/aes.h>

EVP_CIPHER_CTX* ctx = NULL;
unsigned char my_key[256] = {0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x0, 0x1};
unsigned char initial_vector[128] = {0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0x10};

void decrypt_block(const unsigned char* ciphertext, unsigned char* plaintext, const unsigned char* key, const unsigned char* iv) {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv);

    int len;
    EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, AES_BLOCK_SIZE);

    EVP_CIPHER_CTX_free(ctx);
}

// Function to decrypt a range of blocks
void decrypt_blocks(
    const unsigned char *ciphertext,
    unsigned char *plaintext,
    const unsigned char* key,
    const unsigned char* iv,
    int start,
    int end
) {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();

    GGML_ASSERT(EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv) == 1);

    int len;
    GGML_ASSERT(EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, end - start) == 1);

    if ((end - start) % AES_BLOCK_SIZE) {
        GGML_ASSERT(EVP_DecryptFinal_ex(ctx, plaintext + len, &len) == 1);
    }

    EVP_CIPHER_CTX_free(ctx);
}

const size_t BLOCK_SIZE = 4096;
thread_local unsigned char out[BLOCK_SIZE + 32];

std::atomic<int64_t> decrypt_time = 0;
std::atomic<size_t> decrypt_size = 0;

void aes_256_ecb_decrypt(void *dst, const void *src, size_t count) {
    auto start = get_micro();
    unsigned char fake_iv[AES_BLOCK_SIZE];
    decrypt_blocks((const unsigned char *)src, out, my_key, fake_iv, 0, count);
    memcpy(dst, src, count);
#ifdef TZ_LLM_MEASURE
    decrypt_time += get_micro() - start;
    decrypt_size += count;
#endif
}

const char FAKE_KEY = 0;
void fake_decrypt(void *dst, const void *src, size_t count) {
    for (size_t i = 0; i < count; i++) {
        ((char *)dst)[i] = ((const char *)src)[i] ^ FAKE_KEY;
    }
}

struct decrypt_unit {
    void *dst;
    const void *src;
    size_t count;

    decrypt_unit(void *dst, const void *src, size_t count)
        : dst(dst), src(src), count(count) {}
};

struct decrypt_state {
    void *dst;
    const void *src;
    size_t count;
    std::mutex lock;
    std::queue<decrypt_unit> unit_queue;
    std::atomic<int> unfinished;

    decrypt_state(void *dst, const void *src, size_t count)
        : dst(dst), src(src), count(count) {
        for (size_t i = 0; i < count; i += BLOCK_SIZE) {
            size_t remain = count - i;
            if (remain > BLOCK_SIZE)
                remain = BLOCK_SIZE;
            unit_queue.emplace((char *)dst + i, (char *)src + i, remain);
        }
        this->unfinished = unit_queue.size();
    }

    std::optional<decrypt_unit> next_unit(void) {
        std::lock_guard<std::mutex> _(this->lock);
        if (unit_queue.empty()) {
            return std::nullopt;
        } else {
            auto unit = unit_queue.front();
            unit_queue.pop();
            return unit;
        }
    }

    bool is_finished(void) {
        return this->unfinished == 0;
    }
};

static void do_decrypt(decrypt_unit &unit, std::shared_ptr<decrypt_state> state) {
    // fake_decrypt(unit.dst, unit.src, unit.count);
    aes_256_ecb_decrypt(unit.dst, unit.src, unit.count);
    GGML_ASSERT(state->unfinished > 0);
    state->unfinished--;
}

void crypto_mgr::decrypt_step(void) {
    std::shared_ptr<decrypt_state> state;
    std::optional<decrypt_unit> unit;

    {
        std::lock_guard<std::mutex> _(task_queue_lock);
        if (task_queue.empty()) {
            unit = std::nullopt;
        } else {
            state = task_queue.front();
            unit = state->next_unit();
            if (!unit.has_value()) {
                task_queue.pop();
            }
        }
    }
    if (!unit.has_value()) {
        return;
    }

    do_decrypt(unit.value(), state);
}

void crypto_mgr::launch_decrypt(void *dst, const void *src, size_t count) {
    std::lock_guard<std::mutex> _(decrypt_states_lock);
    if (decrypt_states.find(dst) != decrypt_states.end()) return;
    GGML_ASSERT(decrypt_states.find(dst) == decrypt_states.end());
    auto state = std::make_shared<decrypt_state>(dst, src, count);
    decrypt_states.emplace(dst, state);
    task_queue.push(state);
}

void crypto_mgr::wait_decrypt(void *dst) {
    std::shared_ptr<decrypt_state> state;
    {
        std::lock_guard<std::mutex> _(decrypt_states_lock);
        auto iter = decrypt_states.find(dst);
        GGML_ASSERT(iter != decrypt_states.end());
        state = iter->second;
    }
    while (true) {
        if (state->unfinished == 0) {
            return;
        }
        auto unit = state->next_unit();
        if (unit.has_value()) {
            do_decrypt(unit.value(), state);
        }
    }
}

void crypto_mgr::finish_decrypt(void *dst) {
    std::lock_guard<std::mutex> _(decrypt_states_lock);
    auto iter = decrypt_states.find(dst);
    GGML_ASSERT(iter != decrypt_states.end());
    decrypt_states.erase(iter);
}
