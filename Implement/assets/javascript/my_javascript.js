/**
 * Created by eric on 3/24/16.
 */

function draw_rect(node){
    var d = $("#map_select")[0].value == "map_grid" ? 0 : 406;
    x = node.attr("x")*30-d;
    y = node.attr("y")*30-d;
    var context = $("#map")[0].getContext("2d");
    context.fillStyle='lightgray';
//    context.strokeStyle = 'black';
    context.lineWidth = 1;

    context.strokeRect(x, y, 12, 12);
    context.fillRect(x, y, 12, 12);
    context.fillStyle = "black";
    context.fillText(node.attr("item")[0] == null ? " " : node.attr("item")[0].toUpperCase(), x+3, y+10);
}

function draw_arrow(x, y) {
    $("#start")[0].innerText = x +","+  y;
    canvas = $("#map")[0];
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    map_name = $("#map_select")[0].value;
    draw_map(map_name);
    var d = map_name == "map_grid" ? 0 : 406;
    x = x * 30 - d - 9;
    y = y * 30 - d - 24;
    //Loading of the home test image - img1
    var img1 = new Image();


    //drawing of the test image - img1
    img1.onload = function () {
        //draw background image
        ctx.drawImage(img1, x, y, 30, 30);

        //draw a box over the top
        // ctx.fillStyle = "rgba(200, 0, 0, 0.5)";
        // ctx.fillRect(0, 0, 500, 500);
    };

    img1.src = 'assets/images/pin.png';
}

color_dict = {  "blue"      :"60b5e6",
                "brick"     :"b32723",
                "concrete"  :"918f90",
                "flower"    :"e9168c",
                "grass"     :"18a554",
                "gravel"    :"6f6c9b",
                "wood"      :"f4854f",
                "yellow"    :"fcdc41",
                "tower"     :"d7926b",
                "butterfly" :"66cbdd",
                "fish"      :"ee37a1"};

mark = {
    ""          :"0,0,0,0,0,0,",
    "barstool"  :"1,0,0,0,0,0,",
    "chair"     :"0,1,0,0,0,0,",
    "easel"     :"0,0,1,0,0,0,",
    "hatrack"   :"0,0,0,1,0,0,",
    "lamp"      :"0,0,0,0,1,0,",
    "sofa"      :"0,0,0,0,0,1,",
    "blue"      :"1,0,0,0,0,0,0,0,",
    "brick"     :"0,1,0,0,0,0,0,0,",
    "concrete"  :"0,0,1,0,0,0,0,0,",
    "flower"    :"0,0,0,1,0,0,0,0,",
    "grass"     :"0,0,0,0,1,0,0,0,",
    "gravel"    :"0,0,0,0,0,1,0,0,",
    "wood"      :"0,0,0,0,0,0,1,0,",
    "yellow"    :"0,0,0,0,0,0,0,1,",
    "tower"     :"1,0,0,",
    "butterfly" :"0,1,0,",
    "fish"      :"0,0,1,"};

function draw_edge(edge){
    node1 = edge.attr("node1");
    node2 = edge.attr("node2");
    var context = $("#map")[0].getContext("2d");
    var d = $("#map_select")[0].value == "map_grid" ? -6 : 400;
    x1 = node1.split(",")[0]*30-d;
    y1 = node1.split(",")[1]*30-d;
    x2 = node2.split(",")[0]*30-d;
    y2 = node2.split(",")[1]*30-d;
    context.beginPath();
    context.lineWidth = 12;
    context.fillStyle = eval("color_dict." + edge.attr("wall"));
    context.strokeStyle = color_dict[edge.attr("wall")];//eval("color_dict." + edge.attr("floor"));
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
    context.closePath();

    context.beginPath();
    context.lineWidth = 8;
    context.fillStyle = eval("color_dict." + edge.attr("floor"));
    context.strokeStyle = color_dict[edge.attr("floor")];//eval("color_dict." + edge.attr("floor"));
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
    context.closePath();
}

function draw_map(map_name){
    $.get('assets/maps/'+ map_name +'.xml', function (d) {

        var nodes = $(d).find('node');

        // draw edges
        var edges = $(d).find("edge");
        edges.each(function(){
            draw_edge($(this));
        });

        // draw nodes
        nodes.each(function(){
            var node = $(this);
            draw_rect(node);
        });
    });
}
var map_xml;
function get_vector(map_name, x, y, dir) {
    $.get('assets/maps/' + map_name + '.xml', function (d) {
        var result = "";
        map_xml = d;
        directions = [[[1, 0], [-1, 0], [0, -1], [0, 1]], //0
                      [[0, -1], [0, 1], [-1, 0], [1, 0]], //90
                      [[-1, 0], [1, 0], [0, 1], [-1, 0]], //180
                      [[0, 1], [0, -1], [1, 0], [-1, 0]]]; //270

        //node
        //self
        var node = $(d).find('node[x=' + x + '][y=' + y + ']');
        result += node.size() == 1 ? mark[node.attr("item")] : "0,0,0,0,0,0,";
//        result += ',';
//        alert(node.size());

        for (i = 0; i < 4; i++) {
        // nodes in four directions
            node = $(d).find('node[x=' + (parseInt(x) + directions[dir/90][i][0]) + ']' +
                             '[y=' + (parseInt(y) + directions[dir/90][i][1]) + ']');
//            result += node.size() == 1 ? mark[node.attr("item")] : "000000";
            node1 = x + y;
            node2 = (parseInt(x) + directions[dir/90][i][0]) + ',' + (parseInt(y) + directions[dir/90][i][1]);
            edge = $(d).find('edge[node1="' + node1 + '"][node2="' + node2 + '"]');
            if (edge.size() == 0) {
                edge = $(d).find('edge[node1="' + node2 + '"][node2="' + node1 + '"]');
            }
            if (node.size() == 1 && edge.size() == 1) {
                result += mark[node.attr("item")];
            }
            else
                result += "0,0,0,0,0,0,";
//            result += ',';
        }

        for (i = 0; i < 4; i++) {
            node1 = x + ',' + y;
            node2 = (parseInt(x) + directions[dir/90][i][0]) + ',' + (parseInt(y) + directions[dir/90][i][1]);
            edge = $(d).find('edge[node1="' + node1 + '"][node2="' + node2 + '"]');
            if (edge.size() == 0) {
                edge = $(d).find('edge[node1="' + node2 + '"][node2="' + node1 + '"]');
            }

            result += edge.size() == 1 ? mark[edge.attr("floor")] : "0,0,0,0,0,0,0,0,";
            result += edge.size() == 1 ? mark[edge.attr("wall")] : "0,0,0,";

        }
        $("#matrix").append('[').append(result).append('],<br/>');
//        $("#matrix").append(result);
    });
//    return result;
}

function get_matrix_of(map_name) {
    var matrix = $("#matrix");
    for (i = 0; i <25; i++) {
//        matrix.append('[');
        for (j = 0; j < 25; j++) {
//            matrix.append('[');
            for (k = 0; k < 4; k++) {
//                matrix.append('[');
                get_vector(map_name, i, j, k*90);
//                matrix.append(']');
            }
//            matrix.append(']');
        }
//        matrix.append(']');
    }
//    matrix.append("end of loop");
}

function getPosition(event)
{
    var x = event.x;
    var y = event.y;

    var canvas = $("#map")[0];
    x -= canvas.offsetLeft - document.body.scrollLeft;
    y -= canvas.offsetTop - document.body.scrollTop;
    map_name = $("#map_select")[0].value;
    var d = map_name == "map_grid" ? 0 : 406;
    x = parseInt((x+d)/30);
    y = parseInt((y+d)/30);

    $.get('assets/maps/' + map_name + '.xml', function (d) {
        var node = $(d).find('node[x=' + x + '][y=' + y + ']');
        if (node.size() == 0) {
            alert("start point invalid!");
        }
        else {
            draw_arrow(x, y);
        }
    });

}

$(document).ready(function(){

    draw_map("map_one");
    var canvas = $("#map")[0];
    canvas.addEventListener("mousedown", getPosition, false);

    $("#map_select").change(function(){
        $("#start")[0].innerText = "?,?";
        var canvas = $("canvas")[0];
        var context = $("#map")[0].getContext("2d");
        context.clearRect(0, 0, canvas.width, canvas.height);
        draw_map($("#map_select")[0].value);
    });

    $("#judge").click(function(){
        var start_point = $("#start")[0].innerText.split(',');
        var instruction = $("#instruction").val();
        $.get("/judge/",{'instruction':instruction}, function(ret){
            $('#result').html(ret)
        });
    });
    $('#dict').click(function(){
        $.getJSON('ajax_dict',function(ret){
            $.each(ret, function(i,item){
                $('#dict_result').append(i+' '+item + '<br>');
                // i 为索引，item为遍历值
            });
        });
    });
    $('#list').click(function(){
        $.getJSON('ajax_list',function(ret){
            //返回值 ret 在这里是一个列表
            for (var i = ret.length - 1; i >= 0; i--) {
                // 把 ret 的每一项显示在网页上
                $('#list_result').append(' ' + ret[i])
            };
        });
    });
});
