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
#include "kpu.h"
#include <platform.h>
#include <printf.h>
#include <string.h>
#include <stdlib.h>
#include <sysctl.h>
#include "uarths.h"
#include "mobilenetv1.h"

#define PLL0_OUTPUT_FREQ 800000000UL
#define PLL1_OUTPUT_FREQ 300000000UL

volatile uint32_t g_ai_done_flag;

#define IN_SIZE 224

#define OUT_SIZE 7 * 7 * 1024
#define FEATURE_SIZE    1000
volatile uint32_t g_cnt_layer;
uint8_t g_kpu_outbuf[OUT_SIZE] __attribute__((aligned(128)));
float features[FEATURE_SIZE];
extern const unsigned char gImage_image[] __attribute__((aligned(128)));
kpu_task_t task;

static int ai_done(void *ctx)
{
    g_ai_done_flag = 1;
    return 0;
}

int ai_step(void *ctx)
{
    switch (g_cnt_layer)
    {
    case 0 ... 25:
        kpu_conv2d(task.layers + g_cnt_layer);
        break;
    case 26:
        kpu_conv2d_output(task.layers + g_cnt_layer, 5, g_kpu_outbuf, ai_step, NULL);
        break;
	case 27:
	{
		quantize_param_t q1 = { .scale = 0.0163548170351515,.bias = 0 }, q2 = { .scale = 0.0021889562700309,.bias = 0 };
		kpu_global_average_pool(g_kpu_outbuf, &q1, 7*7, 1024, AI_IO_BASE_ADDR + task.layers[g_cnt_layer].image_addr.data.image_src_addr * 64, &q2);
		kpu_matmul_begin(task.layers + g_cnt_layer, 5, (uint64_t *)g_kpu_outbuf, ai_done, NULL);
		break;
	}
    default:
        printf("Unexpcted.\n");
        while (1);
        break;
    }

    printf("%d\n", g_cnt_layer);
    g_cnt_layer++;
    return 0;
}

int main()
{
    /* Set CPU and dvp clk */
    sysctl_pll_set_freq(SYSCTL_PLL0, PLL0_OUTPUT_FREQ);
    sysctl_pll_set_freq(SYSCTL_PLL1, PLL1_OUTPUT_FREQ);
    sysctl_clock_enable(SYSCTL_CLOCK_AI);
    uarths_init();
    plic_init();
    sysctl_enable_irq();
    
    kpu_task_mobilenetv1_init(&task);

    size_t j;
    for (j = 0; j < 1; j++)
    {
        g_cnt_layer = 0;
        g_ai_done_flag = 0;
        kpu_init(task.eight_bit_mode, ai_step, NULL);
        /* start to calculate */
        uint64_t t1 = sysctl_get_time_us();
        kpu_input_with_padding(task.layers, gImage_image, IN_SIZE, IN_SIZE, 3);
        sysctl_disable_irq();
	    ai_step(NULL);
        sysctl_enable_irq();
        while (!g_ai_done_flag){}

		quantize_param_t q = { .scale = task.output_scale,.bias = task.output_bias };

        kpu_matmul_end(g_kpu_outbuf, FEATURE_SIZE, features, &q);
        t1 = sysctl_get_time_us() - t1;
        printf("takes %f ms\n", t1 / 1000.0);
        size_t i;

        for (i = 0; i < FEATURE_SIZE; i++)
            printf("%f, ", features[i]);
        printf("done\n");
    }
    while (1)
        ;
}