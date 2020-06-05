import React from 'react';
import * as d3 from 'd3'
import './App.css';
import ChartEngine from './components/chartEngine';


function App() {
  d3.selectAll('document');
  return (
    <div className="App">
      <ChartEngine width={1000} height={600}/>
    </div>
  );
}

export default App;
