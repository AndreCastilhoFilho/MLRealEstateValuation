using MlNetRealState.App;
using MlNetRealState.App.Models;

class Program
{
    static readonly string DataPath = "data/real_estate.csv";
    static readonly string ModelPath = "models/real_estate_model.zip";

    static void Main()
    {
        var _realEstateModel = new RealEstateModel();

        var model = _realEstateModel.TrainModel(DataPath);
        var sampleData = new RealEstateData { Location = 3, Rooms = 4, Bathrooms = 2, SquareMeters = 80, YearBuilt = 2010 };
        var predictedPrice = _realEstateModel.PredictPrice(model, sampleData);

        Console.WriteLine($"Model trained and saved successfully. {predictedPrice}");
    }
}