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

#import <UIKit/UIKit.h>

@interface MNISTDeepCNN: NSObject
+ (void)initialize;
- (int)forward: (void *)ptrImage;
- (void)close;
@end
