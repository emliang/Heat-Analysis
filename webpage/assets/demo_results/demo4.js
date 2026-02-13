/**
 * TD-ACOPF results data
 * TODO: Replace placeholder imageUrl with actual result images
 */
var opfResultsData = [
    {
      imageUrl: 'images/results/opf_result_1.png',
      title: 'Convergence Profile',
      description: 'Iterative convergence of the resistance-temperature feedback loop showing branch impedance updates across OPF iterations.'
    },
    {
      imageUrl: 'images/results/opf_result_2.png',
      title: 'Load Shedding Map',
      description: 'Spatially-resolved load shedding and branch congestion under heatwave conditions from the TD-ACOPF solution.'
    }
  ];
  
  // Function to show OPF results
  function showOPFResults() {
    ResultsViewer.showResults('TD-ACOPF Results', opfResultsData);
  }
