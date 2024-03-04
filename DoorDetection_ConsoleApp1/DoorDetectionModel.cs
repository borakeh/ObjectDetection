using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static DoorDetection_ConsoleApp1.DoorDetection;

namespace DoorDetection_ConsoleApp1
{
    internal class DoorDetectionModel
    {
        private readonly ITransformer _model;
        private readonly PredictionEngine<ModelInput, ModelOutput> _predictionEngine;

        public DoorDetectionModel(string modelPath)
        {
            var mlContext = new MLContext();
            _model = mlContext.Model.Load(modelPath, out var modelSchema);
            _predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_model);
        }

        public ModelOutput Predict(ModelInput input)
        {
            return _predictionEngine.Predict(input);
        }
    }
}
