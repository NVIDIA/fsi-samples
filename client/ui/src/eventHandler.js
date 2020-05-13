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
        //console.log(d3.event);
        that.mouse = { x, y };
        that.mousePage = { 'x': d3.event.clientX, 'y': d3.event.clientY };
        let point = null;
        if (that.starting) {
            if (that.starting.groupName === "outputs") {
                point = that.translateCorr(that.starting.from, true);
            }
            else {
                point = that.translateCorr(that.starting.from, false);
            }
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
    return function (d) {
        that.tooltip.transition()
            .delay(30)
            .duration(200)
            .style("opacity", 1);
//        that.tooltip.html(JSON.stringify(d))
        that.tooltip.html(JSON.stringify(d))
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
        that.tooltip.transition()
            .duration(100)
            .style("opacity", 0);
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
