import * as d3 from 'd3';
import { Chart } from './chart';
/**
 * @this string
 */
export function handleClicked(that: Chart) {
  return function(d: any): void {
    // this should be svg dom
    // that should be the class instance
    if (
      d3.event.target.tagName === 'rect' ||
      d3.event.target.tagName === 'text'
    ) {
      const nodeDatum = d3.select(d3.event.target.parentNode).datum();
      that.setState({
        opacity: 1,
        x: that.mousePage.x,
        y: that.mousePage.y,
        nodeX: that.mouse.x,
        nodeY: that.mouse.y,
        addMenu: false,
        nodeDatum: nodeDatum
      });
    } else {
      const [x, y] = d3.mouse(this);
      that.mouse = { x, y };
      that.starting = null;
      that.setState({ opacity: 0, x: -1000, y: -1000 });
    }
  };
}

export function handleMouseMoved(that: Chart) {
  return function(d: any): void {
    let [x, y] = d3.mouse(this);
    that.mouse = { x, y };
    that.mousePage = { x: d3.event.clientX, y: d3.event.clientY };
    let point = null;
    if (that.starting) {
      point = that.starting.point;
    }
    const transform = d3.zoomTransform(this);
    //console.log('m', x, y);
    [x, y] = transform.invert([x, y]);
    //console.log('m_p', that.mousePage.x, that.mousePage.y);
    //console.log('t_m', x, y);
    that.mouseLink = that.mouseLink
      .data(that.starting ? ['ab'] : [])
      .join('line')
      .attr('x1', that.starting && point.x)
      .attr('y1', that.starting && point.y)
      .attr('x2', that.mouse && x)
      .attr('y2', that.mouse && y);
  };
}

export function handleMouseLeft(that: Chart) {
  return function(d: any): void {
    this.mouse = null;
  };
}

export function handleMouseOver(that: Chart): Function {
  function constructTable(d: any): string {
    const key = Object.keys(d)[0];
    let header = `<div>Port: ${key.split('.')[1]}</div>`;
    const portType = d[key].portType;
    header += `<div>Port Type:${portType}</div>`;
    const columnObj = d[key].content;
    const columnKeys = Object.keys(columnObj);
    if (columnKeys.length > 0) {
      header += '<table>';
      header += '<tr>';
      header += '<th>Column Name</th>';
      for (let i = 0; i < columnKeys.length; i++) {
        header += `<th>${columnKeys[i]}</th>`;
      }
      header += '</tr>';
      header += '<tr>';
      header += '<th>Type</th>';
      for (let i = 0; i < columnKeys.length; i++) {
        header += `<td>${columnObj[columnKeys[i]]}</td>`;
      }
      header += '</tr>';
      header += '</table>';
    }
    return header;
  }

  return function(d: any): void {
    that.tooltip
      .transition()
      .delay(30)
      .duration(200)
      .style('opacity', 1);
    //        that.tooltip.html(JSON.stringify(d))
    const transform = d3.zoomTransform(this);
    const [x, y] = transform.invert([that.mouse.x, that.mouse.y]);
    console.log('non transformed', that.mouse.x, that.mouse.y);
    console.log('transformed', x, y);
    that.tooltip
      .html(constructTable(d))
      //      .style('left', that.mouse.x + 25 + 'px')
      //     .style('top', that.mouse.y + 'px');
      .style('left', that.mousePage.x + 25 + 'px')
      .style('top', that.mousePage.y + 'px');
    //add this
    const selection = d3.select(this);
    selection
      .transition()
      .delay(20)
      .duration(200)
      .attr('r', that.circleRadius + 1)
      .style('opacity', 1);
  };
}

export function handleMouseOut(that: Chart) {
  return function(d: any): void {
    that.tooltip
      .style('opacity', 0)
      .style('left', -1000 + 'px')
      .style('top', -1000 + 'px');
    //add this
    const selection = d3.select(this);
    selection
      .transition()
      .delay(20)
      .attr('r', that.circleRadius)
      .duration(200);
  };
}

export function handleHighlight(that: Chart, color: string, style: string) {
  return function(d: any): void {
    d3.select(this).style('cursor', style);
    d3.select(this.parentNode).attr('stroke', color);
  };
}

export function handleDeHighlight(that: Chart) {
  return function(d: any): void {
    d3.select(this).style('cursor', 'default');
    d3.select(this.parentNode).attr('stroke', null);
  };
}

export function handleRightClick(that: Chart) {
  return function(d: any): void {
    const transform = d3.zoomTransform(this);
    const [x, y] = transform.invert([that.mouse.x, that.mouse.y]);
    d3.event.preventDefault();
    that.setState({
      opacity: 1,
      x: that.mousePage.x,
      y: that.mousePage.y,
      nodeX: x,
      nodeY: y,
      addMenu: true
    });
  };
}

export function handleEdit(that: Chart) {
  return function(d: any): void {
    //        that.setState({'opacity':1, 'x':that.mousePage.x, 'y':that.mousePage.y, 'nodeX': that.mouse.x, 'nodeY': that.mouse.y, 'addMenu': false});
  };
}
