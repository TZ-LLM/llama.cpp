#include <mutex>
#include <queue>
#include <unordered_map>
#include <memory>

struct decrypt_state;

struct crypto_mgr {
    std::mutex task_queue_lock;
    std::queue<std::shared_ptr<decrypt_state>> task_queue;
    std::mutex decrypt_states_lock;
    std::unordered_map<void *, std::shared_ptr<decrypt_state>> decrypt_states;

    void launch_decrypt(void *dst, const void *src, size_t count);
    void wait_decrypt(void *dst);
    void finish_decrypt(void *dst);
    void decrypt_step(void);
};
