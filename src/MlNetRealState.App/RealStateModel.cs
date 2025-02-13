using Microsoft.ML;
using Microsoft.ML.Transforms;
using MlNetRealState.App.Models;

namespace MlNetRealState.App
{
    public class RealEstateModel
    {
        public readonly MLContext MlContext;

        public RealEstateModel()
        {
            MlContext = new MLContext(seed: 0);
        }

        /// <summary>
        /// Creates an ML.NET pipeline with preprocessing steps and the FastTree trainer.
        /// </summary>
        public IEstimator<ITransformer> CreatePipeline(int numberOfTrees = 700, double learningRate = 0.02)
        {
            var dataProcessPipeline = MlContext.Transforms.ReplaceMissingValues(nameof(RealEstateData.Rooms), replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
                .Append(MlContext.Transforms.ReplaceMissingValues(nameof(RealEstateData.Bathrooms), replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(MlContext.Transforms.ReplaceMissingValues(nameof(RealEstateData.SquareMeters), replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(MlContext.Transforms.ReplaceMissingValues(nameof(RealEstateData.YearBuilt), replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(MlContext.Transforms.Concatenate("Features",
                    nameof(RealEstateData.Rooms),
                    nameof(RealEstateData.Bathrooms),
                    nameof(RealEstateData.SquareMeters),
                    nameof(RealEstateData.YearBuilt)))
                .Append(MlContext.Transforms.NormalizeMeanVariance("Features"))
                // Apply log transformation directly to Label
                .Append(MlContext.Transforms.Expression("Label", "(x) => log(x)", "Price"));

            var trainer = MlContext.Regression.Trainers.FastTree(
                labelColumnName: "Label",
                featureColumnName: "Features",
                numberOfLeaves: 50,
                minimumExampleCountPerLeaf: 5,
                numberOfTrees: numberOfTrees,
                learningRate: learningRate);

            return dataProcessPipeline.Append(trainer);
        }

        public IDataView LoadData(string dataPath)
        {
            return MlContext.Data.LoadFromTextFile<RealEstateData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');
        }

        public (ITransformer Model, IDataView TestData) TrainModelWithSplit(string dataPath)
        {
            var data = LoadData(dataPath);
            var trainTestSplit = MlContext.Data.TrainTestSplit(data, testFraction: 0.3);
            var model = CreatePipeline().Fit(trainTestSplit.TrainSet);
            return (model, trainTestSplit.TestSet);
        }

        public ITransformer TrainModel(string dataPath)
        {
            return TrainModelWithSplit(dataPath).Model;
        }

        public float PredictPrice(ITransformer model, RealEstateData inputData)
        {
            var predictionEngine = MlContext.Model.CreatePredictionEngine<RealEstateData, RealEstatePrediction>(model);
            var prediction = predictionEngine.Predict(inputData);
            // Reverse log transformation
            return (float)Math.Exp(prediction.Price);
        }

        public double PerformCrossValidation(string dataPath)
        {
            var dataView = LoadData(dataPath);
            var pipeline = CreatePipeline();

            var cvResults = MlContext.Regression.CrossValidate(
                data: dataView,
                estimator: pipeline,
                numberOfFolds: 5);

            return cvResults.Average(r => r.Metrics.RootMeanSquaredError);
        }

        public List<double> TrackConvergence(string dataPath, int maxTrees = 100, int step = 10)
        {
            var rmseList = new List<double>();
            var data = LoadData(dataPath);
            var trainTestSplit = MlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            for (int trees = step; trees <= maxTrees; trees += step)
            {
                var pipeline = CreatePipeline(numberOfTrees: trees);
                var model = pipeline.Fit(trainTestSplit.TrainSet);
                var predictions = model.Transform(trainTestSplit.TestSet);
                var metrics = MlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

                rmseList.Add(metrics.RootMeanSquaredError);
                Console.WriteLine($"Trees: {trees}, RMSE = {metrics.RootMeanSquaredError}");
            }

            return rmseList;
        }

        public (ITransformer Model, IDataView TestData, double RMSE) TrainAndEvaluateModel(string dataPath)
        {
            var data = LoadData(dataPath);
            var trainTestSplit = MlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var model = CreatePipeline().Fit(trainTestSplit.TrainSet);

            var predictions = model.Transform(trainTestSplit.TestSet);
            var metrics = MlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            return (model, trainTestSplit.TestSet, metrics.RootMeanSquaredError);
        }
    }
}
