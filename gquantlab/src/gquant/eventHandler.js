import * as d3 from 'd3';

export function handleClicked(that) {
    return function (d) {
        // this should be svg dom 
        // that should be the class instance
        if (d3.event.target.tagName === 'rect' || d3.event.target.tagName === 'text'){
            let nodeDatum = d3.select(d3.event.target.parentNode).datum();
            that.setState({'opacity':1, 'x':that.mousePage.x, 'y':that.mousePage.y, 'nodeX': that.mouse.x, 'nodeY': that.mouse.y, 'addMenu': false, 'nodeDatum': nodeDatum});
        }
        else{
            const [x, y] = d3.mouse(this);
            that.mouse = { x, y };
            that.starting = null;
            that.setState({ 'opacity': 0, 'x': -1000, 'y': -1000 });
        }
    }
}

export function handleMouseMoved(that) {
    return function (d) {
        let [x, y] = d3.mouse(this);
        that.mouse = { x, y };
        that.mousePage = { 'x': d3.event.clientX, 'y': d3.event.clientY };
        let point = null;
        if (that.starting) {
            point = that.starting.point;
        }
        let transform = d3.zoomTransform(this);
        [x, y] = transform.invert([x, y]);
        that.mouseLink = that.mouseLink
            .data(that.starting ? ['ab'] : [])
            .join("line")
            .attr("x1", that.starting && point.x)
            .attr("y1", that.starting && point.y)
            .attr("x2", that.mouse && x)
            .attr("y2", that.mouse && y);
        }
}

export function handleMouseLeft(that) {
    return function (d) {
        this.mouse = null;
    }
}

export function handleMouseOver(that) {
    function constructTable(d) {
        let key = Object.keys(d)[0];
        let header = `<div>Port: ${key.split(".")[1]}</div>`;
        let portType = d[key].portType;
        header += `<div>Port Type:${portType}</div>`;
        let columnObj = d[key].content;
        let columnKeys = Object.keys(columnObj);
        if (columnKeys.length > 0) {
            header += "<table>";
            header += "<tr>";
            header += "<th>Column Name</th>";
            for (let i = 0; i < columnKeys.length; i++) {
                header += `<th>${columnKeys[i]}</th>`;
            }
            header += "</tr>";
            header += "<tr>";
            header += "<th>Type</th>";
            for (let i = 0; i < columnKeys.length; i++) {
                header += `<td>${columnObj[columnKeys[i]]}</td>`;
            }
            header += "</tr>";
            header += "</table>";
        }
        return header;
    }

    return function (d) {
        that.tooltip.transition()
            .delay(30)
            .duration(200)
            .style("opacity", 1);
//        that.tooltip.html(JSON.stringify(d))
        that.tooltip.html(constructTable(d))
            .style("left", (d3.event.pageX + 25) + "px")
            .style("top", (d3.event.pageY) + "px");
        //add this
        const selection = d3.select(this);
        selection
            .transition()
            .delay("20")
            .duration("200")
            .attr("r", that.circleRadius + 1)
            .style("opacity", 1);
    }
}

export function handleMouseOut(that) {
    return function (d) {
        that.tooltip
            .style("opacity", 0)
            .style("left", (-1000) + "px")
            .style("top", (-1000) + "px");
        //add this
        const selection = d3.select(this);
        selection
            .transition()
            .delay("20")
            .attr("r", that.circleRadius)
            .duration("200");
    }
}

export function handleHighlight(that, color, style) {
    return function (d) {
    d3.select(this).style('cursor', style);
    d3.select(this.parentNode).attr("stroke", color);
    }
}

export function handleDeHighlight(that) {
    return function (d) {
        d3.select(this).style('cursor', 'default');
        d3.select(this.parentNode).attr("stroke", null);
    }
}

export function handleRightClick(that){
    return function(d){
        let transform = d3.zoomTransform(this);
        let [x, y] = transform.invert([that.mouse.x, that.mouse.y]);
        d3.event.preventDefault();
        that.setState({'opacity':1, 'x':that.mousePage.x, 'y':that.mousePage.y, 'nodeX': x, 'nodeY': y, 'addMenu': true});
    }
}

export function handleEdit(that){
    return function(d){
//        that.setState({'opacity':1, 'x':that.mousePage.x, 'y':that.mousePage.y, 'nodeX': that.mouse.x, 'nodeY': that.mouse.y, 'addMenu': false});
    }
}
