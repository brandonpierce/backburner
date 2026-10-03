// gpuwarm US SECONDS: a 32-thread Metal dispatch every US microseconds (not waited for), like phone-attn's PA_GPU_WARM_US
#import <Metal/Metal.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
int main(int argc, char ** argv) {
    @autoreleasepool {
        int us = argc > 1 ? atoi(argv[1]) : 1000; double secs = argc > 2 ? atof(argv[2]) : 60;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError * e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@"kernel void w(device uint * x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 1u; }" options:nil error:&e];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"w"] error:&e];
        id<MTLBuffer> buf = [dev newBufferWithLength:256 options:MTLResourceStorageModePrivate];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        struct timespec t0, t; clock_gettime(CLOCK_MONOTONIC, &t0);
        long n = 0;
        for (;;) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso]; [enc setBuffer:buf offset:0 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                [enc endEncoding]; [cb commit]; n++;
            }
            usleep(us);
            clock_gettime(CLOCK_MONOTONIC, &t);
            if ((t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) * 1e-9 > secs) break;
        }
        fprintf(stderr, "gpuwarm: %ld dispatches\n", n);
    }
    return 0;
}
