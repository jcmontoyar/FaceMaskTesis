// Import dependencies
import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";

import "./App.css";
// 2. TODO - Import drawing utility here

import { drawRect } from "./utilities";
const blazeface = require('@tensorflow-models/blazeface');

const classes = ["FC", "FP", "NF" ]

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  
  // Main function
  const runCoco = async () => {
    // 3. TODO - Load network 
    // e.g. const net = await cocossd.load();
    
    const net = await tf.loadLayersModel("https://tensorflowrealtimefacemask.s3.us-south.cloud-object-storage.appdomain.cloud/model.json");
    const modelFaces = await blazeface.load();

    //  Loop and detect hands
    setInterval(() => {

      detectFaces(modelFaces, net);
    }, 16.7);
  };

  const detectFaces = async (modelFaces, net) =>{
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // Draw mesh
      const ctx = canvasRef.current.getContext("2d");
      const img = tf.browser.fromPixels(video);

      // Pass in an image or video to the model. The model returns an array of
      // bounding boxes, probabilities, and landmarks, one for each detected face.

      const returnTensors = false; // Pass in `true` to get tensors back, rather than values.
      const predictions = await modelFaces.estimateFaces(img, returnTensors);

      let pixels = video.getBoundingClientRect();

      if (predictions.length > 0 && predictions.length < 2) {
        
        /*
        `predictions` is an array of objects describing each detected face, for example:

        [
          {
            topLeft: [232.28, 145.26],
            bottomRight: [449.75, 308.36],
            probability: [0.998],
            landmarks: [
              [295.13, 177.64], // right eye
              [382.32, 175.56], // left eye
              [341.18, 205.03], // nose
              [345.12, 250.61], // mouth
              [252.76, 211.37], // right ear
              [431.20, 204.93] // left ear
            ]
          }
        ]
        */

     
        const start = predictions[0].topLeft;
        const end = predictions[0].bottomRight;
        const size = [end[0] - start[0], end[1] - start[1]];  
        
        console.log("One face found")
        const resized = tf.image.resizeBilinear(img, [224, 224]).div(tf.scalar(255))

        const cast = tf.cast(resized, "float32")
        const expanded = cast.expandDims(0);
       

        const pred = net.predict(expanded).dataSync()

        let fc = pred[0];
        let fp = pred[1];
        let nf = pred[2];

        let color = "red"
        let text = ""
        let value = 1.0
        if(fc > fp && fc > nf){
          value = fc
          color = "blue";
          text = classes[0]
        }
        else if(fp > fc && fp > nf){
          value = fp
          color = "yellow";
          text = classes[1]
        }
        else{
          value = nf
          color = "red";
          text = classes[2]
        }
        requestAnimationFrame(()=>{
          // Render a rectangle over each detected face.
          // Set styling
          ctx.strokeStyle = color
          ctx.lineWidth = 10
          ctx.fillStyle = 'white'
          ctx.font = '30px Arial'

          ctx.beginPath()
          ctx.fillText(text + ' - ' + Math.round(value*100)/100, size[0], size[1]-10)
          ctx.rect(start[0], start[1], size[0], size[1]);
          ctx.stroke()
        })
        // Display the winner

        tf.dispose(img);
        tf.dispose(resized);
        tf.dispose(cast);
        tf.dispose(expanded);
 

       }
      
      tf.dispose(img);
    }
  }

  const detect = async (net) => {
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // Draw mesh
      const ctx = canvasRef.current.getContext("2d");

      // 4. TODO - Make Detections
      const img = tf.browser.fromPixels(video);
      const resized = tf.image.resizeBilinear(img, [224, 224]).toFloat();
      // Normalize the image 

 

      const expanded = resized.expandDims(0);

      const pred = net.predict(expanded).dataSync()
      console.log(pred)

      //const boxes = await obj[1].array()
      //const classes = await obj[2].array()
      //const scores = await obj[4].array()
     

     

      // 5. TODO - Update drawing utility
      //requestAnimationFrame(()=>{drawRect(boxes[0], classes[0], scores[0], 0.8, videoWidth, videoHeight, ctx)}); 

      // free memory
      tf.dispose(img);
      tf.dispose(resized);
  
      tf.dispose(expanded);
      tf.dispose(pred);

    }
  };

  useEffect(()=>{runCoco()},[]);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          muted={true} 
          style={{
            position: "absolute",

            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 224,
            height: 224,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
    
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 8,
            width: 224,
            height: 224,
          }}
        />
      </header>
    </div>
  );
}

export default App;
