/**
 * Demand calibration results data
 * TODO: Replace placeholder imageUrl with actual result images
 */
var demandResultsData = [
    {
      imageUrl: 'images/results/demand_result_1.png',
      title: 'Temperature-Demand Relationship',
      description: 'Calibrated BAIT thermal-comfort model showing the nonlinear relationship between temperature and electricity demand per country.'
    },
    {
      imageUrl: 'images/results/demand_result_2.png',
      title: 'Demand Profile Validation',
      description: 'Comparison of modelled vs. observed ENTSO-E hourly demand during historical heatwave periods.'
    }
  ];
  
  // Function to show demand calibration results
  function showDemandResults() {
    ResultsViewer.showResults('Demand Calibration Results', demandResultsData);
  }
