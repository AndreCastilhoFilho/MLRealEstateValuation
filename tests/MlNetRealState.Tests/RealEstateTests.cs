using Microsoft.ML;
using Microsoft.ML.Data;
using MlNetRealState.App;
using System.Diagnostics;

namespace MlNetRealState.Tests
{
    public class RealEstateTests
    {
        private readonly RealEstateModel _realEstateModel;
        private readonly string _dataPath = "data/real_estate.csv";

        public RealEstateTests()
        {
            _realEstateModel = new RealEstateModel();
        }

        [Fact]
        public void Model_ShouldTrainSuccessfully()
        {
            var model = _realEstateModel.TrainModel("data/real_estate.csv");
            Assert.NotNull(model);
        }

        [Fact]
        public void Model_ShouldPredictReasonablePrice()
        {
            var model = _realEstateModel.TrainModel(_dataPath);
            var sampleData = new RealEstateData { Location = 3, Rooms = 4, Bathrooms = 2, SquareMeters = 80, YearBuilt = 2010 };
            var predictedPrice = _realEstateModel.PredictPrice(model, sampleData);

            Assert.True(predictedPrice > 0, "Predicted price should be greater than zero.");
        }

        [Fact]
        public void Model_ShouldHaveAcceptableRMSE()
        {
            var rmse = _realEstateModel.PerformCrossValidation(_dataPath);
            Console.WriteLine($"Cross-Validation RMSE: {rmse}");
            Assert.True(rmse < 100000, "RMSE should be below 100,000 for acceptable accuracy.");
        }

        [Fact]
        public void Model_ShouldHandleExtremeValues()
        {
            var model = _realEstateModel.TrainModel(_dataPath);
            var sampleData = new RealEstateData { Location = 100, Rooms = 20, Bathrooms = 10, SquareMeters = 1000, YearBuilt = 1800 };
            var predictedPrice = _realEstateModel.PredictPrice(model, sampleData);

            Assert.True(predictedPrice > 0 && predictedPrice < 10_000_000, "Predicted price should be in a reasonable range.");
        }

        [Fact]
        public void Model_ShouldPerformWellOnTestData()
        {
            var (model, testData) = _realEstateModel.TrainModelWithSplit(_dataPath);
            var predictions = model.Transform(testData);

            // Extract actual prices
            var actualPrices = testData.GetColumn<float>("Price").ToArray();

            //  Reverse log transformation on predicted values
            var predictedPrices = predictions.GetColumn<float>("Score")
                .Select(p => (float)Math.Exp(p)) // Convert back to normal price scale
                .ToArray();

            var metrics = _realEstateModel.MlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

            Console.WriteLine($"Test RMSE: {metrics.RootMeanSquaredError}");

            // Recalculate RMSE after applying exp()
            double adjustedRMSE = Math.Sqrt(actualPrices.Zip(predictedPrices, (actual, pred) => Math.Pow(actual - pred, 2)).Average());
            Console.WriteLine($"Adjusted RMSE (after exp transform): {adjustedRMSE}");

            PlotActualVsPredicted(actualPrices, predictedPrices);
        }


        [Fact]
        public void Model_ShouldIdentifyImportantFeatures()
        {
            var model = _realEstateModel.TrainModel(_dataPath);
            var dataView = _realEstateModel.LoadData(_dataPath);
            var transformedData = model.Transform(dataView);
            var permutationMetrics = _realEstateModel.MlContext.Regression.PermutationFeatureImportance(
                model, transformedData, labelColumnName: "Label", permutationCount: 50);


            var featureImportance = permutationMetrics
                .Select((metric, index) => new { Feature = index, Importance = metric.Value.RSquared.Mean })
                .OrderByDescending(f => f.Importance)
                .ToList();

            foreach (var feature in featureImportance)
            {
                Console.WriteLine($"Feature {feature.Feature}: Importance {feature.Importance}");
            }

            Assert.True(Math.Abs(featureImportance.First().Importance) > 0.01, "At least one feature should have measurable importance.");

        }
        [Fact]
        public void Model_ShouldNotOverfit()
        {
            var (model, testData) = _realEstateModel.TrainModelWithSplit(_dataPath);

            // Load the original dataset for training evaluation
            var trainData = _realEstateModel.LoadData(_dataPath);

            // Evaluate on training data
            var trainPredictions = model.Transform(trainData);
            var trainMetrics = _realEstateModel.MlContext.Regression.Evaluate(trainPredictions, labelColumnName: "Label");
            Console.WriteLine($"Training RMSE: {trainMetrics.RootMeanSquaredError}");

            // Evaluate on test data
            var testPredictions = model.Transform(testData);
            var testMetrics = _realEstateModel.MlContext.Regression.Evaluate(testPredictions, labelColumnName: "Label");
            Console.WriteLine($"Test RMSE: {testMetrics.RootMeanSquaredError}");

            // Ensure that test RMSE is not significantly worse than training RMSE
            Assert.True(testMetrics.RootMeanSquaredError < trainMetrics.RootMeanSquaredError * 1.5,
                "Test RMSE should not be significantly higher than training RMSE (overfitting check).");
        }
        [Fact]
        public void Model_ShouldConverge()
        {
            var rmseList = _realEstateModel.TrackConvergence(_dataPath, maxTrees: 100, step: 10);
            PlotConvergence(rmseList);

            // Ensure RMSE is decreasing over time
            int decreasingSteps = 0;
            for (int i = 1; i < rmseList.Count; i++)
            {
                if (rmseList[i] < rmseList[i - 1]) // Count steps where RMSE decreased
                    decreasingSteps++;
            }

            double decreaseRatio = (double)decreasingSteps / (rmseList.Count - 1);
            Console.WriteLine($"Decrease Ratio: {decreaseRatio:P2}");

            Assert.True(decreaseRatio >= 0.7, "RMSE should decrease in at least 70% of the iterations.");
        }

        public void PlotConvergence(List<double> rmseList)
        {
            var plt = new ScottPlot.Plot();

            double[] iterations = Enumerable.Range(1, rmseList.Count).Select(i => (double)i).ToArray();
            double[] rmseValues = rmseList.ToArray();

            var scatter = plt.Add.Scatter(iterations, rmseValues);
            scatter.LegendText = "RMSE per Iteration";

            plt.Title("Model Convergence");
            plt.XLabel("Iteration");
            plt.YLabel("RMSE");
            plt.Legend.IsVisible = true;

            // Save the plot
            string filePath = "convergence_plot.png";
            plt.SavePng(filePath, 600, 400);
            Console.WriteLine($"Convergence plot saved as {filePath}");

            // Open the plot automatically (Windows)
            try
            {
                Process.Start(new ProcessStartInfo(filePath) { UseShellExecute = true });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Could not open plot: {ex.Message}");
            }
        }

        public void PlotActualVsPredicted(ITransformer model, IDataView testData)
        {
            var mlContext = new MLContext();

            // Get predictions
            var predictions = model.Transform(testData);

            // Extract actual and predicted values using "Price" instead of "Label"
            var actualPrices = testData.GetColumn<float>("Price").ToArray();
            var predictedPrices = predictions.GetColumn<float>("Score").ToArray(); // ML.NET stores predictions in "Score"

            // Create a ScottPlot graph
            var plt = new ScottPlot.Plot();

            // Add scatter plot for actual vs. predicted values
            var scatter = plt.Add.Scatter(actualPrices, predictedPrices);
            scatter.LegendText = "Predicted vs. Actual";

            // Add reference line (perfect prediction)
            var line = plt.Add.Line(0, 0, actualPrices.Max(), actualPrices.Max());
            line.LegendText = "Perfect Prediction (y = x)";

            plt.Title("Actual vs. Predicted Prices");
            plt.XLabel("Actual Price");
            plt.YLabel("Predicted Price");
            plt.Legend.IsVisible = true;

            // Save the plot
            plt.SavePng("actual_vs_predicted.png", 600, 400);
            Console.WriteLine("Plot saved as actual_vs_predicted.png");
            // Automatically open the image (Windows only)
            string filePath = "actual_vs_predicted.png";
            try
            {
                Process.Start(new ProcessStartInfo(filePath) { UseShellExecute = true });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Could not open plot: {ex.Message}");
            }
        }

        public void PlotActualVsPredicted(float[] actualPrices, float[] predictedPrices)
        {
            var plt = new ScottPlot.Plot();
            plt.Add.Scatter(actualPrices, predictedPrices).LegendText = "Predicted vs. Actual";
            plt.Add.Line(0, 0, actualPrices.Max(), actualPrices.Max()).LegendText = "Perfect Prediction (y = x)";
            plt.Title("Actual vs. Predicted Prices");
            plt.XLabel("Actual Price");
            plt.YLabel("Predicted Price");
            plt.Legend.IsVisible = true;

            string filePath = "actual_vs_predicted.png";
            plt.SavePng(filePath, 600, 400);
            Console.WriteLine($"Plot saved as {filePath}");
            try
            {
                Process.Start(new ProcessStartInfo(filePath) { UseShellExecute = true });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Could not open plot: {ex.Message}");
            }

        }
    }
}