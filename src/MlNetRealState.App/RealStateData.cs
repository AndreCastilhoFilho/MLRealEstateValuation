using Microsoft.ML.Data;

namespace MlNetRealState.App
{
    public class RealEstateData
    {
        [LoadColumn(0)] public float Location { get; set; }
        [LoadColumn(1)] public float Rooms { get; set; }
        [LoadColumn(2)] public float Bathrooms { get; set; }
        [LoadColumn(3)] public float SquareMeters { get; set; }
        [LoadColumn(4)] public float YearBuilt { get; set; }
        [LoadColumn(5)] public float Price { get; set; }
    }

}
