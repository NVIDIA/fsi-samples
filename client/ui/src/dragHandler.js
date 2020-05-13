import * as d3 from 'd3';

export function drag(setState, that) {
  var offsetX = 0;
  var offsetY = 0;

  function dragstarted(d) {
    //if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    //d3.select(this).attr("stroke", "black");
    d3.select(this.parentNode).attr("stroke", "black");
    offsetX = d3.event.x;
    offsetY = d3.event.y;
  }

  function dragged(d) {
    const [x, y] = d3.mouse(this.parentNode.parentNode);
    d3.select(this.parentNode).raise().attr("transform", (d, i) => `translate(${d.x = x - offsetX}, ${d.y = y - offsetY})`)
    that.link = that.link
      .data(that.edgeData())
      .join('line')
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

  }

  function dragended(d) {
    d3.select(this.parentNode).attr("stroke", null);
    setState({ "nodes": that.bars.data() });
  }

  return d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}
