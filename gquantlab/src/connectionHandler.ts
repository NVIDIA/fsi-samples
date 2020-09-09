import * as d3 from 'd3';
import { validConnection } from './validator';
import { IEdge } from './document';
import { Chart } from './chart';

export function handleMouseUp(that: Chart) {
  return function(d: any): void {
    console.log('mouse up');
    d3.event.stopPropagation();
    d3.event.preventDefault();
    let groupName = d3.select(this.parentNode).attr('group');
    if (that.starting) {
      const datum: any = d3.select(this.parentNode).datum();
      const toId = datum.id;
      groupName = d3.select(this.parentNode).attr('group');
      const toPort = Object.keys(d)[0].split('.')[1];
      if (groupName === 'outputs') {
        if (that.starting.groupName === 'inputs') {
          const newEdge = {
            from: toId + '.' + toPort,
            to: that.starting.from
          };
          const position = that.props.edges.findIndex(
            (d: IEdge) => d.from === newEdge.from && d.to === newEdge.to
          );
          if (position >= 0) {
            that.props.edges.splice(position, 1);
          } else {
            // only make the connection if it is valid
            if (validConnection(that)(newEdge.from, newEdge.to)) {
              that.props.edges.push(newEdge);
            }
          }
          that.connectionUpdate();
          // const jsonString = JSON.stringify(output);
          // that.updateInputs(jsonString);
          // that.drawLinks();
          that.isDirty = false;
          //that.props.setChartState({'edges': that.props.edges})
          //let links = that.edgeData();
          //link = that.link.data(links)
          //    .join("line");
        }
      } else {
        if (that.starting.groupName === 'outputs') {
          const newEdge = {
            from: that.starting.from,
            to: toId + '.' + toPort
          };
          const position = that.props.edges.findIndex(
            (d: IEdge) => d.from === newEdge.from && d.to === newEdge.to
          );
          if (position >= 0) {
            that.props.edges.splice(position, 1);
          } else {
            // only make the connection if it is valid
            if (validConnection(that)(newEdge.from, newEdge.to)) {
              that.props.edges.push(newEdge);
            }
          }
          that.connectionUpdate();
          // const jsonString = JSON.stringify(output);
          // that.updateInputs(jsonString);
          // that.drawLinks();
          that.isDirty = false;
          //that.props.setChartState({'edges': that.props.edges})
          // links = edges.map(edge_map);
          // link = link.data(links)
          //     .join("line");
        }
      }
      that.starting = null;
    }
    that.drawCircles();
  };
}

export function handleMouseDown(that: Chart) {
  return function(d: any): void {
    console.log('mouse down');
    d3.event.stopPropagation();
    d3.event.preventDefault();
    let groupName = d3.select(this.parentNode).attr('group');
    const portStr = Object.keys(d)[0];

    //if (!that.starting) {
    const datum: any = d3.select(this.parentNode).datum();
    const fromId = datum.id;
    groupName = d3.select(this.parentNode).attr('group');
    const fromPort = Object.keys(d)[0].split('.')[1];
    that.starting = {
      from: fromId + '.' + fromPort,
      groupName: groupName,
      point: null
    };

    // }
    if (groupName === 'inputs') {
      //inputs, to in edges
      const index = that.props.edges.findIndex((d: IEdge) => d.to === portStr);
      if (index >= 0) {
        that.starting = {
          from: that.props.edges[index].from,
          groupName: 'outputs',
          point: null
        };
        that.props.edges.splice(index, 1);
        // const output = that.configFile();
        // const jsonString = JSON.stringify(output);
        that.drawLinks();
        that.isDirty = true;
        // that.updateInputs(jsonString);
      }
    } else {
      //outputs, from in edges
      const index = that.props.edges.findIndex(
        (d: IEdge) => d.from === portStr
      );
      if (index >= 0) {
        that.starting = {
          from: that.props.edges[index].to,
          groupName: 'inputs',
          point: null
        };
        that.props.edges.splice(index, 1);
        //const output = that.configFile();
        //const jsonString = JSON.stringify(output);
        that.drawLinks();
        //that.updateInputs(jsonString);
        that.isDirty = true;
        //that.props.setChartState({'edges': that.props.edges})
        // links = edges.map(edge_map);
        // link = link.data(links)
        //     .join("line");
      }
    }
    if (that.starting.groupName === 'outputs') {
      that.starting['point'] = that.translateCorr(that.starting.from, true);
    } else {
      that.starting['point'] = that.translateCorr(that.starting.from, false);
    }

    that.drawCircles();
  };
}
