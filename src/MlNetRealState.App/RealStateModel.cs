using Microsoft.ML;
using Microsoft.ML.Transforms;
using SkiaSharp;

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
        /// <param name="numberOfTrees">Number of trees for the FastTree trainer.</param>
        /// <param name="learningRate">Learning rate for the FastTree trainer.</param>
        /// <returns>A configured estimator pipeline.</returns>
        private IEstimator<ITransformer> CreatePipeline(int numberOfTrees = 700, double learningRate = 0.02)
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

                // 🔥 Apply log transformation to Label (Price)
                .Append(MlContext.Transforms.CustomMapping<RealEstateData, TransformedRealEstateData>(
                    (input, output) => { output.Label = (float)Math.Log(input.Price); }, contractName: "LogTransform"));

            var trainer = MlContext.Regression.Trainers.FastTree(
                labelColumnName: "Label",
                featureColumnName: "Features",
                numberOfLeaves: 50,
                minimumExampleCountPerLeaf: 5,
                numberOfTrees: numberOfTrees,
                learningRate: learningRate);

            return dataProcessPipeline.Append(trainer);
        }



        /// <summary>
        /// Loads the real estate data from a CSV file.
        /// </summary>
        /// <param name="dataPath">The path to the CSV data file.</param>
        /// <returns>An IDataView containing the loaded data.</returns>
        public IDataView LoadData(string dataPath)
        {
            return MlContext.Data.LoadFromTextFile<RealEstateData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');
        }


        /// <summary>
        /// Trains the model using a train/test split.
        /// </summary>
        /// <param name="dataPath">The path to the CSV data file.</param>
        /// <returns>A tuple containing the trained model and the test data.</returns>
        public (ITransformer Model, IDataView TestData) TrainModelWithSplit(string dataPath)
        {
            var data = LoadData(dataPath);
            var trainTestSplit = MlContext.Data.TrainTestSplit(data, testFraction: 0.3);
            var model = CreatePipeline().Fit(trainTestSplit.TrainSet);
            return (model, trainTestSplit.TestSet);
        }

        /// <summary>
        /// Trains the model on the full dataset.
        /// </summary>
        /// <param name="dataPath">The path to the CSV data file.</param>
        /// <returns>The trained model.</returns>
        public ITransformer TrainModel(string dataPath)
        {
            return TrainModelWithSplit(dataPath).Model;
        }

        /// <summary>
        /// Predicts the price of a house given its features.
        /// Note: The PredictionEngine is not thread-safe.
        /// </summary>
        /// <param name="model">The trained model.</param>
        /// <param name="inputData">A single instance of real estate data.</param>
        /// <returns>The predicted price.</returns>
        public float PredictPrice(ITransformer model, RealEstateData inputData)
        {
            var predictionEngine = MlContext.Model.CreatePredictionEngine<RealEstateData, RealEstatePrediction>(model);
            var prediction = predictionEngine.Predict(inputData);

            // 🔥 Reverse the log transformation (convert log(price) back to normal scale)
            return (float)Math.Exp(prediction.Price);
        }

        /// <summary>
        /// Performs 5-fold cross-validation on the dataset.
        /// </summary>
        /// <param name="dataPath">The path to the CSV data file.</param>
        /// <returns>The average RMSE from cross-validation.</returns>
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


        /// <summary>
        /// Tracks convergence by varying the number of trees and evaluating the RMSE.
        /// </summary>
        /// <param name="dataPath">The path to the CSV data file.</param>
        /// <param name="maxTrees">Maximum number of trees to test.</param>
        /// <param name="step">Step increment for the number of trees.</param>
        /// <returns>A list of RMSE values for each tested number of trees.</returns>
        public List<double> TrackConvergence(string dataPath, int maxTrees = 100, int step = 10)
        {
            var rmseList = new List<double>();
            var data = LoadData(dataPath);
            var trainTestSplit = MlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            for (int trees = step; trees <= maxTrees; trees += step)
            {
                var pipeline = CreatePipeline(numberOfTrees: trees);

                var model = pipeline.Fit(trainTestSplit.TrainSet);

                // Evaluate RMSE on the same test set
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

            // Evaluate the model on test data
            var predictions = model.Transform(trainTestSplit.TestSet);
            var metrics = MlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            return (model, trainTestSplit.TestSet, metrics.RootMeanSquaredError);
        }


    }
}
