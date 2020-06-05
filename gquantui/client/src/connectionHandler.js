import * as d3 from 'd3';

export function handleMouseUp(that) {
    return function (d) {
        console.log('mouse up');
        d3.event.stopPropagation();
        d3.event.preventDefault();
        let groupName = d3.select(this.parentNode).attr('group');
        if (that.starting) {
            let toId = d3.select(this.parentNode).datum().id;
            groupName = d3.select(this.parentNode).attr('group');
            let toPort =  Object.keys(d)[0].split('.')[1];
            if (groupName === "outputs") {
                if (that.starting.groupName === "inputs") {
                    let newEdge = {
                        "from": toId + "." + toPort,
                        "to": that.starting.from
                    };
                    let position = that.props.edges.findIndex((d) => d.from === newEdge.from && d.to === newEdge.to);
                    console.log(newEdge, position);
                    if (position >= 0) {
                        that.props.edges.splice(position, 1);
                    }
                    else {
                        that.props.edges.push(newEdge);
                    }
                    let output = that.configFile();
                    let jsonString = JSON.stringify(output);
                    that.drawLinks();
                    that.updateInputs(jsonString);
                    //that.props.setChartState({'edges': that.props.edges})
                    //let links = that.edgeData();
                    //link = that.link.data(links)
                    //    .join("line");
                }
            }
            else {
                if (that.starting.groupName === "outputs") {
                    let newEdge = {
                        "from": that.starting.from,
                        "to": toId + "." + toPort
                    };
                    let position = that.props.edges.findIndex((d) => d.from === newEdge.from && d.to === newEdge.to);
                    if (position >= 0) {
                        that.props.edges.splice(position, 1);
                    }
                    else {
                        that.props.edges.push(newEdge);
                    }
                    let output = that.configFile();
                    let jsonString = JSON.stringify(output);
                    that.drawLinks();
                    that.updateInputs(jsonString);
                    //that.props.setChartState({'edges': that.props.edges})
                    // links = edges.map(edge_map);
                    // link = link.data(links)
                    //     .join("line");
                }
            }
            that.starting = null;
        }
    }
}

export function handleMouseDown(that){
    return function (d){
        console.log('mouse down');
        d3.event.stopPropagation();
        d3.event.preventDefault();
        let groupName = d3.select(this.parentNode).attr('group');
        let portStr = Object.keys(d)[0];

        //if (!that.starting) {
            let fromId = d3.select(this.parentNode).datum().id;
            groupName = d3.select(this.parentNode).attr('group');
            let fromPort = Object.keys(d)[0].split('.')[1];
            that.starting = { 'from': fromId + "." + fromPort, "groupName": groupName };
       // }
        if (groupName === "inputs") {
            //inputs, to in edges
            let index = that.props.edges.findIndex((d) => d.to === portStr);
            if (index >= 0) {

                that.starting = { 'from': that.props.edges[index].from, "groupName": "outputs" };
                that.props.edges.splice(index, 1);
                    let output = that.configFile();
                    let jsonString = JSON.stringify(output);
                    that.drawLinks();
                    that.updateInputs(jsonString);

                //that.props.setChartState({'edges': that.props.edges})
                // links = edges.map(edge_map);
                // link = link.data(links)
                //     .join("line");
            }
        }
        else {
            //outputs, from in edges
            let index = that.props.edges.findIndex((d) => d.from === portStr);
            if (index >= 0) {
                that.starting = { 'from': that.props.edges[index].to, "groupName": "inputs" };
                that.props.edges.splice(index, 1);
                    let output = that.configFile();
                    let jsonString = JSON.stringify(output);
                    that.drawLinks();
                    that.updateInputs(jsonString);
                //that.props.setChartState({'edges': that.props.edges})
                // links = edges.map(edge_map);
                // link = link.data(links)
                //     .join("line");
            }
        }
    }
}
