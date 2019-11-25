/* Copyright 2018 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>
#include <sysctl.h>
#include <string.h>
#include "uarths.h"
#include "kpu.h"
#include "incbin.h"
#include "iomem.h"
#include "syscalls.h"

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX

#define PLL0_OUTPUT_FREQ 800000000UL
#define PLL1_OUTPUT_FREQ 400000000UL

INCBIN(model, "mobilenetv1_1.0.kmodel");

kpu_model_context_t task;

volatile uint32_t g_ai_done_flag;

extern const unsigned char gImage_image[] __attribute__((aligned(128)));

#define IMAGE_DATA_SIZE (224 * 224 * 3)
uint8_t *pImage;

static int ai_done(void *ctx)
{
    g_ai_done_flag = 1;
    return 0;
}

int main()
{
    /* Set CPU and KPU clk */
    sysctl_pll_set_freq(SYSCTL_PLL0, PLL0_OUTPUT_FREQ);
    sysctl_pll_set_freq(SYSCTL_PLL1, PLL1_OUTPUT_FREQ);
    sysctl_clock_enable(SYSCTL_CLOCK_AI);
    
    uarths_init();
    plic_init();
    
    pImage = (uint8_t*)iomem_malloc(IMAGE_DATA_SIZE);
    if (pImage)
    {
        memcpy(pImage, gImage_image, IMAGE_DATA_SIZE);
    }
    else
    {
        printf("Bad allocation!\n");
        return 1;
    }

    if (kpu_load_kmodel(&task, model_data) != 0)
    {
        printf("\nmodel init error\n");
        return -1;
    }

    sysctl_enable_irq();

    printf("System Start\n");
    g_ai_done_flag = 0;
    kpu_run_kmodel(&task, pImage, DMAC_CHANNEL5, ai_done, NULL);
    while (g_ai_done_flag == 0);
    float *output;
    size_t output_size;
    kpu_get_output(&task, 0, (uint8_t **)&output, &output_size);

    for (uint32_t i = 0; i < output_size / (sizeof(float)); i++)
        printf("%f ", output[i]);
    printf("\ndone\n");
    
    while (1);
}