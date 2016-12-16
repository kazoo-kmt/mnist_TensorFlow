//
//  MetalCNNBasicViewController.swift
//  mnist_TensorFlow
//  Created by Kazu Komoto on 12/05/16.
//  Copyright © 2016 Kazu Komoto. All rights reserved.
/*
    The following code is based on MetalCNNBasicViewController.swift by Apple and Shuichi Tsusumi
*/

import UIKit

class MetalCNNBasicViewController: UIViewController {
  
    // Networks we have
    var network: MNISTDeepCNN!
    
    // MNIST dataset image parameters
    let mnistInputWidth  = 28
    let mnistInputHeight = 28
    
    @IBOutlet private weak var digitView: DrawView!
    @IBOutlet private weak var predictionLabel: UILabel!
    @IBOutlet private weak var clearBtn: UIButton!
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        clearBtn.isHidden = true
        predictionLabel.text = nil

        network = MNISTDeepCNN()
    }
    
    @IBAction func clearBtnTapped(sender: UIButton) {
        // clear the digitview
        digitView.lines = []
        digitView.setNeedsDisplay()
        predictionLabel.text = nil
        clearBtn.isHidden = true
    }
    
    @IBAction func detectBtnTapped(sender: UIButton) {
        // get the digitView context so we can get the pixel values from it to intput to network
        let context = digitView.getViewContext()
        
        // validate NeuralNetwork was initialized properly
        assert(network != nil)
        
        // run the network forward pass
        let label = network.forward(context!.data!)
      
        // show the prediction
        predictionLabel.text = "\(label)"
        clearBtn.isHidden = false
    }
  
  override func viewDidDisappear(_ animated: Bool) {
    network.close()
  }
}
