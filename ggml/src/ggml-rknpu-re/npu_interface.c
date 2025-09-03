/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * This file is part of rk3588-npu.
 *
 * rk3588-npu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rk3588-npu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rk3588-npu.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "rknpu-ioctl.h"
#include "npu_hw.h"
#include "npu_interface.h"

#include <pthread.h>

int fd;
pthread_once_t fd_once;
int npu_open(void);

void fd_init(void)
{
  fd = npu_open();
  printf("%s %d: fd %d\n", __func__, __LINE__, fd);
}

void* mem_allocate(size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint64_t *handle) {
  pthread_once(&fd_once, fd_init);
  int ret;
  struct rknpu_mem_create mem_create = {
    0,
    .flags = flags | RKNPU_MEM_NON_CACHEABLE,
    .size = size,
    0,0,0
  };

  ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create);
  if(ret < 0)  {
    printf("RKNPU_MEM_CREATE failed %d\n",ret);
    return NULL;
  }

  struct rknpu_mem_map mem_map = { .handle = mem_create.handle, 0, .offset=0 };
  ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map);
  if(ret < 0) {
    printf("RKNPU_MEM_MAP failed %d\n",ret);
    return NULL;
  }

  void *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem_map.offset);
  if (map == MAP_FAILED) {
    printf("Error: mmap failed, len=%zu, offset=%llu, errno %d\n", size, mem_map.offset, errno);
    return NULL;
  }

  *dma_addr = mem_create.dma_addr;
  *obj = mem_create.obj_addr;
  *handle = mem_create.handle;
  static uint64_t sum = 0;
  sum += size;
  // printf("%s %d sum %ldMB dma %p va %p\n", __func__, __LINE__, sum / 1024 / 1024, (void *)mem_create.dma_addr, map);
  return map;
}

void mem_destroy(void *addr, size_t len, uint64_t handle, uint64_t obj_addr) {
  pthread_once(&fd_once, fd_init);
  munmap(addr, len);

  int ret;
  struct rknpu_mem_destroy destroy = {
    .handle = handle ,
    0,
    .obj_addr = obj_addr
  };

  ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &destroy);
  if (ret <0) {
    printf("RKNPU_MEM_DESTROY failed %d\n",ret);
  }
}

int npu_open(void) {

  char buf1[256], buf2[256], buf3[256];

  memset(buf1, 0 ,sizeof(buf1));
  memset(buf2, 0 ,sizeof(buf2));
  memset(buf3, 0, sizeof(buf3));

  // Open DRI called "rknpu"
  int fd = open("/dev/dri/card1", O_RDWR);
  if(fd<0) {
    printf("Failed to open /dev/dri/card1 %d\n",errno);
    return fd;
  }

  struct drm_version dv;
  memset(&dv, 0, sizeof(dv));
  dv.name = buf1;
  dv.name_len = sizeof(buf1);
  dv.date = buf2;
  dv.date_len = sizeof(buf2);
  dv.desc = buf3;
  dv.desc_len = sizeof(buf3);

  int ret = ioctl(fd, DRM_IOCTL_VERSION, &dv);
  if (ret <0) {
    printf("DRM_IOCTL_VERISON failed %d\n",ret);
    return ret;
  }
  printf("drm name is %s - %s - %s\n", dv.name, dv.date, dv.desc);
  return fd;
}

int npu_reset(void) {
  pthread_once(&fd_once, fd_init);

  // Reset the NPU
  struct rknpu_action act = {
    .flags = RKNPU_ACT_RESET,
    0,
  };
  return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);	
}

int npu_submit(__u64 task_obj_addr, __u32 core_mask)
{
  pthread_once(&fd_once, fd_init);
  struct rknpu_submit submit = {
    .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
    .timeout = 6000,
    .task_start = 0,
    .task_number = 1,
    .task_counter = 0,
    .priority = 0,
    .task_obj_addr = task_obj_addr,
    .regcfg_obj_addr = 0,
    .task_base_addr = 0,
    .user_data = 0,
    .core_mask = core_mask,
    .fence_fd = -1,
    .subcore_task = {
      {
        .task_start = 0,
        .task_number = 1,
      }, {0, 1}, {0, 1}, {0, 0}, {0, 0}
    },
  };
  return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
}
