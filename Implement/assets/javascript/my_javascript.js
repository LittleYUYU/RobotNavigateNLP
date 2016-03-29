/**
 * Created by eric on 3/24/16.
 */

function draw_rect(node){
    var d = $("select option:selected")[0].value == "map_grid" ? 0 : 406;
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
function draw_edge(edge){
    node1 = edge.attr("node1");
    node2 = edge.attr("node2");
    var context = $("#map")[0].getContext("2d");
    var d = $("select option:selected")[0].value == "map_grid" ? -6 : 400;
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

$(document).ready(function(){
//    $.get('assets/maps/map_one.xml', function (d) {
//        var $map = $(d).find('map');
//        var nodes = $(d).find('node');
//
//        // draw edges
//        var edges = $(d).find("edge");
//        edges.each(function(){
//            draw_edge($(this));
//        });
//
//        // draw nodes
//        nodes.each(function(){
//            var node = $(this);
//            draw_rect(node.attr("x"),node.attr("y"));
//        });
//
//    });
    draw_map("map_one");

    $("#map_select").change(function(){
        var canvas = $("canvas")[0];
        var context = $("#map")[0].getContext("2d");
        context.clearRect(0, 0, canvas.width, canvas.height);
        draw_map($("select option:selected")[0].value);
    });

    $("#judge").click(function(){
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
