using Microsoft.ML.Data;

namespace MlNetRealState.App
{
    public class RealEstatePrediction
    {
        [ColumnName("Score")] public float Price { get; set; }
    }
}
