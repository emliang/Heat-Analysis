/**
 * Conductor thermal model results data
 * TODO: Replace placeholder imageUrl with actual result images
 */
var thermalResultsData = [
    {
      imageUrl: 'images/results/thermal_result_1.png',
      title: 'Heat-Balance Solution',
      description: 'Steady-state conductor temperature computed from the IEEE 738-2012 heat-balance equation under varying ambient conditions.'
    },
    {
      imageUrl: 'images/results/thermal_result_2.png',
      title: 'Dynamic Line Rating',
      description: 'Spatially-resolved ampacity reduction along multi-segment transmission lines during peak heatwave hours.'
    }
  ];
  
  // Function to show thermal model results
  function showThermalResults() {
    ResultsViewer.showResults('Conductor Thermal Model Results', thermalResultsData);
  }
