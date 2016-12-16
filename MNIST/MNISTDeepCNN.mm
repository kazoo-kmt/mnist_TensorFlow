//
//  MNISTDeepCNN.mm
//  mnist_TensorFlow
//
//  Created by Kazu Komoto on 12/05/16.
//  Copyright © 2016 Kazu Komoto. All rights reserved.
//
//
/*
 Deep layer network where we define and encode the correct layers on a command buffer as needed
 This is based on MNISTSingleLayer.swift and MNISTDeepCNN.swift provided by Apple,
 and ViewController.m provided by Matt Rajca.

 MIT License
 
 Copyright (c) 2016 Matt Rajca
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
*/

#import "MNISTDeepCNN.h"

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

using namespace tensorflow;

static constexpr int kImageSide = 28;
static constexpr int kOutputs = 10;
static constexpr int kInputLength = kImageSide * kImageSide;
Session* session;
Status status;

@implementation MNISTDeepCNN

+ (void)initialize
{
  if (self == [MNISTDeepCNN class]) {
    NSLog(@"MNISTDeepCNN is initialized");
    
    status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
      std::cout << status.ToString() << "\n";
      return;
    }
    
    NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"final" ofType:@"pb"];
    
    GraphDef graph;
    status = ReadBinaryProto(Env::Default(), modelPath.fileSystemRepresentation, &graph);
    if (!status.ok()) {
      std::cout << status.ToString() << "\n";
      return;
    }
    
    status = session->Create(graph);
    if (!status.ok()) {
      std::cout << status.ToString() << "\n";
      return;
    }
    
  }
}

- (int)forward: (void *)ptrImage {
  int label = 99;
  
  @autoreleasepool {

  Tensor x(DT_FLOAT, TensorShape({ 1, kInputLength }));

  NSData *imageData = [NSData dataWithBytes:ptrImage length:kInputLength*sizeof(float)];

  for (auto i = 0; i < kInputLength; i++) {
    uint8_t pixel;
    [imageData getBytes:&pixel range:NSMakeRange(i, 1)];
    
    x.matrix<float>().operator()(i) = pixel / 255.0f;
  }

	std::vector<std::pair<string, Tensor>> inputs = {
		{ "x", x }
	};

	const auto start = CACurrentMediaTime();

	std::vector<Tensor> outputs;
	status = session->Run(inputs, {"softmax"}, {}, &outputs);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return -1;
	}

	NSLog(@"Time: %g seconds", CACurrentMediaTime() - start);

	const auto outputMatrix = outputs[0].matrix<float>();

  int bestIndex = -1;
  float bestProbability = 0;
  for (auto i = 0; i < kOutputs; i++) {
    const auto probability = outputMatrix(i);
    if (probability > bestProbability) {
      bestProbability = probability;
      bestIndex = i;
      NSLog(@"%i", bestIndex);
    }
  }
  label = bestIndex;
  }
  return label;
}

- (void)close {
  session->Close();
  NSLog(@"MNISTDeepCNN's session is closed");
}

@end
