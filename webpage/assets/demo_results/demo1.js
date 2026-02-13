/**
 * Heatwave generation results data
 * TODO: Replace placeholder imageUrl with actual result images
 */
var heatwaveResultsData = [
    {
      imageUrl: 'images/results/heatwave_result_1.png',
      title: 'Heatwave Projection Map',
      description: 'Projected heatwave temperature anomalies across Europe using bias-corrected delta mapping from 2019/2022/2024 historical extremes.'
    },
    {
      imageUrl: 'images/results/heatwave_result_2.png',
      title: 'Temporal Evolution',
      description: 'Temporal evolution of morphed heatwave patterns throughout a 7-day extreme event for future scenario (2026-2030).'
    },
    {
      imageUrl: 'images/results/heatwave_result_3.png',
      title: 'Country-Level Comparison',
      description: 'Comparison of peak temperature distributions across ES, FR, IT, DE, GB under different heatwave projection years.'
    }
  ];
  
  // Function to show heatwave results
  function showHeatwaveResults() {
    ResultsViewer.showResults('Heatwave Generation Results', heatwaveResultsData);
  }
