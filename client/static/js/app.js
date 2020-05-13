window.onload = (event) => {
d3.json('/graph').then((d)=>{
  var data = d['nodes'];
  var edges = d['edges'];

  connection_info = {};
  for (var i = 0; i < edges.length; i++){
      children = edges[i].to.split('.')[0];
      parent = edges[i].from.split('.')[0];
      if  (children in connection_info){
          if (connection_info[children].findIndex((d)=>d===parent) < 0){
            connection_info[children].push(parent);
          }
      }
      else{
        connection_info[children] = [parent];
      }
  }

  console.log(connection_info);

  for (var i = 0; i < data.length; i++){
      if (data[i]['id'] in connection_info){
          data[i]['parentIds'] = connection_info[data[i]['id']];
      }
  }

  console.log(data);
  dag = d3.dagStratify()(data);
  d3.sugiyama()
   .size([700, 700])
   .layering(d3.layeringSimplex())
   .decross(d3.decrossOpt())
   .coord(d3.coordVert())(dag);

dag.descendants().forEach((d)=>{d.data['x']=d.x; d.data['y']=d.y; return})



  const tooltip = d3.select("body").append("div")
      .attr("class", "tooltip")
      .style("opacity", 0)
      .style("position", "absolute");

  const svg = d3.select('body').append("svg")
      .attr("width", 1000)
      .attr("height", 1000)
      .attr("font-family", "sans-serif")
      .attr("font-size", "14")
      .attr("text-anchor", "end")
      .on("mouseleave", mouseleft)
      .on("mousemove", mousemoved)
      .on("click", clicked);

var text_height = 25;
var circle_height = 20;
var circle_radius = 8;

var starting = null;
let mouse = null;

  function mouseleft() {
    mouse = null;
  }

  function mousemoved() {
    const [x, y] = d3.mouse(this);
    mouse = {x, y};
    simulation.alpha(0).restart();
  }

   function clicked() {
     console.log('svg clicked');
    mousemoved.call(this);
    starting = null;
    //spawn({x: mouse.x, y: mouse.y});
  }

var mouse_over = function(d) {
        tooltip.transition()
        .delay(30)
        .duration(200)
        .style("opacity", 1);
        tooltip.html(d)
        .style("left", (d3.event.pageX + 25) + "px")
        .style("top", (d3.event.pageY) + "px");
//add this        
        const selection = d3.select(this);
        selection
        .transition()
        .delay("20")
        .duration("200")
        .attr("r", circle_radius+1)
        .style("opacity", 1);
    }

var mouse_out = function(d) {      
        tooltip.transition()        
        .duration(100)      
        .style("opacity", 0);  
//add this        
        const selection = d3.select(this);
        selection
        .transition()
        .delay("20")
        .attr("r", circle_radius)
        .duration("200");
    }

    var mouse_up = function(d){
        console.log('mouse up');
        d3.event.stopPropagation();
        node_id = d3.select(this.parentNode).datum().id;
        group_name = d3.select(this.parentNode).attr('group');
        if(starting){
            to_id = d3.select(this.parentNode).datum().id;
            group_name = d3.select(this.parentNode).attr('group');
            to_port = d;
        if (group_name === "outputs"){
            if (starting.group_name === "inputs"){
                new_edge = {"from": to_id+"."+to_port,
                    "to": starting.from};
                position = edges.findIndex((d)=>d.from===new_edge.from && d.to===new_edge.to);
                if (position >= 0){
                    edges.splice(position, 1);
                }
                else{
                    edges.push(new_edge);
                }
                links = edges.map(edge_map);
                link = link.data(links)
                    .join("line");
            }
        }
       else{
            if (starting.group_name === "outputs"){
                new_edge = {"from": starting.from, 
                    "to": to_id+"."+to_port};
                position = edges.findIndex((d)=>d.from===new_edge.from && d.to===new_edge.to);
                if (position >= 0){
                    edges.splice(position, 1);
                }
                else{
                    edges.push(new_edge);
                }
                links = edges.map(edge_map);
                link = link.data(links)
                    .join("line");
            }
        }
        starting = null;
        simulation.alpha(0).restart();
        }
    }
 
    var mouse_down = function(d){
        console.log('mouse down');
        d3.event.stopPropagation();
        node_id = d3.select(this.parentNode).datum().id;
        group_name = d3.select(this.parentNode).attr('group');
        port_str = node_id+"."+d;

        if (!starting){
            from_id = d3.select(this.parentNode).datum().id;
            group_name = d3.select(this.parentNode).attr('group');
            from_port = d;
            starting = {'from': from_id+"."+from_port, "group_name":group_name};
        }
        if (group_name == "inputs"){
            //inputs, to in edges
            index = edges.findIndex((d)=>d.to===port_str);
            if (index >= 0){
                starting = {'from': edges[index].from, "group_name":"outputs"};
                edges.splice(index, 1);
                links = edges.map(edge_map);
                link = link.data(links)
                    .join("line");
            }
        }
        else{
            //outputs, from in edges
            index = edges.findIndex((d)=>d.from===port_str);
            if (index >= 0){
                starting = {'from': edges[index].to, "group_name":"inputs"};
                edges.splice(index, 1);
                links = edges.map(edge_map);
                link = link.data(links)
                    .join("line");
            }
        }
    }


function translate_corr(port_str, from=false){
      let splits  = port_str.split('.');
      node_id = splits[0];
      output_port = splits[1];
      node_obj = data.filter((d)=>d.id === node_id)[0];
    if (from){
      index = node_obj.outputs.indexOf(output_port);
      let x = node_obj.x + node_obj.width;
      let y = node_obj.y + (index + 0.4)*circle_height + text_height;
      let point = {'x':x, 'y':y, 'id':node_id};
      return  point;
    }
    else{
      index = node_obj.inputs.indexOf(output_port);
      let x = node_obj.x;
      let y = node_obj.y + (index + 0.4)*circle_height + text_height;
      let point = {'x':x, 'y':y, 'id':node_id};
      return  point;
    }
}

function edge_map(d) {
      source_point = translate_corr(d.from, true);
      target_point = translate_corr(d.to, false);
      return {'source':source_point, 'target':target_point};
}

var links = [];

 const simulation = d3.forceSimulation(data)
      .force("charge", d3.forceManyBody().strength(80))
      .force("center", d3.forceCenter(500,500))
      .force("link", d3.forceLink(links));
      //.force("x", d3.forceX())
      //.force("y", d3.forceY());
function drag() {

  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    //d3.select(this).attr("stroke", "black");
    d3.select(this.parentNode).attr("stroke", "black");
  }

  function dragged(d) {
//    d3.select(this).raise().attr("transform", (d, i) => `translate(${d.x = d3.event.x}, ${d.y = d3.event.y})`);
    d3.select(this.parentNode).raise().attr("transform", (d, i) => `translate(${d.x = d3.event.x}, ${d.y = d3.event.y})`);
  }

  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
  //  d3.select(this).attr("stroke", null);
    d3.select(this.parentNode).attr("stroke", null);
    simulation.alpha(0.).restart();
  }

  return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);

    }
 const bar = svg.selectAll("g")
	   .data(data)
       .join("g")
       .attr("transform", (d, i) => `translate(${d.x}, ${d.y})`);
       //

  bar.append("rect")
      .attr("fill", "steelblue")
      .attr("width", (d)=>d.width)
      .attr("height",  (d)=>text_height + Math.max(d.inputs.length, d.outputs.length)*circle_height);

  bar.append("text")
      .attr("fill", "white")
      .attr("x", (d)=> d.width)
      .attr("y", 0)
      .attr("dy", "1.00em")
      .attr("dx", "-1.00em")
      .text((d)=>d.id).call(drag());

 let link = svg.append("g")
      .attr("stroke", "#999")
      .selectAll("line")
      .data(edges.map(edge_map))
      .join('line');

  let mouselink = svg.append("g")
      .attr("stroke", "red")
      .selectAll("line");

var ports_input = bar.append("g")
     .attr("transform", (d, i) => `translate(0, ${text_height})`)
     .attr("group", "inputs");

  ports_input.selectAll("circle")
	  .data((d)=>d.inputs)
      .join("circle")
      .attr("fill", "blue")
      .attr("cx", 0 )
      .attr("cy", (d, i)=> (i + 0.4)*circle_height)
	  .attr("r", circle_radius)
      .on('mouseover', mouse_over)                
      .on("mouseout", mouse_out)
      .on("mousedown", mouse_down)
      .on("mouseup", mouse_up);

var ports_output = bar.append("g")
     .attr("transform", (d, i) => `translate(${d.width}, ${text_height})`)
     .attr("group", "outputs");


  ports_output.selectAll("circle")
      .data((d)=>d.outputs)
      .join("circle")
      .attr("fill", "green")
      .attr("cx", 0)
      .attr("cy", (d, i)=> (i + 0.4)*circle_height)
	  .attr("r", circle_radius)
      .on('mouseover', mouse_over)                
      .on("mouseout", mouse_out)
      .on("mousedown", mouse_down)
      .on("mouseup", mouse_up);

simulation.on("tick", () => {
    bar.attr("transform", (d, i) => `translate(${d.x}, ${d.y})`);
    links = edges.map(edge_map);
    simulation.force("link").links(links);
    link = link.data(links)
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
   //const [x, y] = d3.mouse(this);
   // starting = {'x':mouse.x, 'y':mouse.y, 'from': from_id+"."+from_port, "group_name":group_name};
    if (starting){
        if (starting.group_name == "outputs"){
            point = translate_corr(starting.from, from=true);
        }
        else{
            point = translate_corr(starting.from, from=false);
        }
    }
    else{
        point = null;
    }
    mouselink = mouselink
      .data(starting ? ['ab'] : [])
      .join("line")
        .attr("x1", starting && point.x)
        .attr("y1", starting && point.y)
        .attr("x2", mouse && mouse.x)
        .attr("y2", mouse && mouse.y);
  });

    // window.onclick = (event)=>{
    //     edges.push({"from": "abc_node.output1",
    //                 "to": "efg_node.input1"});
    //     links = edges.map(edge_map);
    //     link = link.data(links)
    //         .join("line");
    //     alert('yes');
    // };

//  invalidation.then(() => simulation.stop());
  //d3.select('body').append(svg.node);
});
};
