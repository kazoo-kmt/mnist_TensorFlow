//
//  ViewController.h
//  MNIST
//
//  Created by Matt on 11/24/16.
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface MNISTDeepCNN: NSObject
- (int)forward: (void *)ptrImage;
@end
