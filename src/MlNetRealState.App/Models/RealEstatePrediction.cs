using Microsoft.ML.Data;

namespace MlNetRealState.App.Models
{
    public class RealEstatePrediction
    {
        [ColumnName("Score")] public float Price { get; set; }
    }
}
