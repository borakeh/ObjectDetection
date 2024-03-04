﻿// This file was auto-generated by ML.NET Model Builder.
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML;
using Microsoft.ML.TorchSharp.AutoFormerV2;
using System.Text.Json.Nodes;
using System.Text.Json;

namespace DoorDetection_ConsoleApp1
{
    public partial class DoorDetection
    {
        public const string RetrainFilePath = @"C:\Users\developer\Desktop\doors\doorsOut\vott-json-export\DoorsWork-export.json";
        public const int TrainingImageWidth = 800;
        public const int TrainingImageHeight = 600;

        /// <summary>
        /// Train a new model with the provided dataset.
        /// </summary>
        /// <param name="outputModelPath">File path for saving the model. Should be similar to "C:\YourPath\ModelName.mlnet"</param>
        /// <param name="inputDataFilePath">Path to the data file for training.</param>
        public static void Train(string outputModelPath, string inputDataFilePath = RetrainFilePath)
        {
           var mlContext = new MLContext();
           mlContext.GpuDeviceId = 0;
           mlContext.FallbackToCpu = false;
           var data = LoadIDataViewFromVOTTFile(mlContext, inputDataFilePath);
           var model = RetrainModel(mlContext, data);
           SaveModel(mlContext, model, data, outputModelPath);
        }

        // <summary>
        /// Load an IDataView from a file path.
        /// </summary>
        /// <param name="mlContext">The common context for all ML.NET operations.</param>
        /// <param name="inputDataFilePath">Path to the vott data file for training.</param>
        public static IDataView LoadIDataViewFromVOTTFile(MLContext mlContext, string inputDataFilePath)
        {
           return mlContext.Data.LoadFromEnumerable(LoadFromVott(inputDataFilePath));
        }

        private static IEnumerable<ModelInput> LoadFromVott(string inputDataFilePath)
        {
            JsonNode jsonNode;
            using (StreamReader r = new StreamReader(inputDataFilePath))
            {
                string json = r.ReadToEnd();
                jsonNode = JsonSerializer.Deserialize<JsonNode>(json);
            }

            var imageData = new List<ModelInput>();
            foreach (KeyValuePair<string, JsonNode> asset in jsonNode["assets"].AsObject())
            {
                var labelList = new List<string>();
                var boxList = new List<float>();

                var sourceWidth = asset.Value["asset"]["size"]["width"].GetValue<float>();
                var sourceHeight = asset.Value["asset"]["size"]["height"].GetValue<float>();   

                CalculateAspectAndOffset(sourceWidth, sourceHeight, TrainingImageWidth, TrainingImageHeight, out float xOffset, out float yOffset, out float aspect);

                foreach (var region in asset.Value["regions"].AsArray())
                {
                    foreach (var tag in region["tags"].AsArray())
                    {
                        labelList.Add(tag.GetValue<string>());
                        var boundingBox = region["boundingBox"];
                        var left = boundingBox["left"].GetValue<float>();
                        var top = boundingBox["top"].GetValue<float>();
                        var width = boundingBox["width"].GetValue<float>();
                        var height = boundingBox["height"].GetValue<float>();

                        boxList.Add(xOffset + (left * aspect));
                        boxList.Add(yOffset + (top * aspect));
                        boxList.Add(xOffset + ((left + width) * aspect));
                        boxList.Add(yOffset + ((top + height) * aspect));
                    }
                    
                }    

                var mlImage = MLImage.CreateFromFile(asset.Value["asset"]["path"].GetValue<string>().Replace("file:", ""));
                var modelInput = new ModelInput()
                {
                    Image = mlImage,
                    Labels = labelList.ToArray(),
                    Box = boxList.ToArray(),
                };    

                imageData.Add(modelInput);
            }

            return imageData;
        }

        private static void CalculateAspectAndOffset(float sourceWidth, float sourceHeight, float destinationWidth, float destinationHeight, out float xOffset, out float yOffset, out float aspect)
        {
            float widthAspect = destinationWidth / sourceWidth;
            float heightAspect = destinationHeight / sourceHeight;
            xOffset = 0;
            yOffset = 0;
            if (heightAspect < widthAspect)
            {
                aspect = heightAspect;
                xOffset = (destinationWidth - (sourceWidth * aspect)) / 2;     
            }    
            else
            {
                aspect = widthAspect;
                yOffset = (destinationHeight - (sourceHeight * aspect)) / 2;
            }
        }

        /// <summary>
        /// Save a model at the specified path.
        /// </summary>
        /// <param name="mlContext">The common context for all ML.NET operations.</param>
        /// <param name="model">Model to save.</param>
        /// <param name="data">IDataView used to train the model.</param>
        /// <param name="modelSavePath">File path for saving the model. Should be similar to "C:\YourPath\ModelName.mlnet.</param>
        public static void SaveModel(MLContext mlContext, ITransformer model, IDataView data, string modelSavePath)
        {
            // Pull the data schema from the IDataView used for training the model
            DataViewSchema dataViewSchema = data.Schema;

            using (var fs = File.Create(modelSavePath))
            {
                mlContext.Model.Save(model, dataViewSchema, fs);
            }
        }


        /// <summary>
        /// Retrain model using the pipeline generated as part of the training process.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainData"></param>
        /// <returns></returns>
        public static ITransformer RetrainModel(MLContext mlContext, IDataView trainData)
        {
            var pipeline = BuildPipeline(mlContext);
            var model = pipeline.Fit(trainData);

            return model;
        }

        /// <summary>
        /// build the pipeline that is used from model builder. Use this function to retrain model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName:@"Labels",inputColumnName:@"Labels",addKeyValueAnnotationsAsText:false)      
                                    .Append(mlContext.Transforms.ResizeImages(outputColumnName:@"Image",inputColumnName:@"Image",imageHeight:TrainingImageHeight,imageWidth:TrainingImageWidth,cropAnchor:ImageResizingEstimator.Anchor.Center,resizing:ImageResizingEstimator.ResizingKind.IsoPad))      
                                    .Append(mlContext.MulticlassClassification.Trainers.ObjectDetection(new ObjectDetectionTrainer.Options(){LabelColumnName=@"Labels",PredictedLabelColumnName=@"PredictedLabel",BoundingBoxColumnName=@"Box",ImageColumnName=@"Image",ScoreColumnName=@"score",MaxEpoch=5,InitLearningRate=1,WeightDecay=0,}))      
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName:@"PredictedLabel",inputColumnName:@"PredictedLabel"));

            return pipeline;
        }
    }
 }
