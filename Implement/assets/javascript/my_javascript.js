/**
 * Created by eric on 3/24/16.
 */

function draw_rect(x, y){
    var context = $("#map")[0].getContext("2d");
    context.fillStyle='#FF000';
    context.strokeRect(x*30-400, y*30-400, 10, 10);
}

function draw_edge(node1, node2){
    
    context.fillStyle='#FF000';
    context.strokeRect(x*30-400, y*30-400, 10, 10);
}

$(document).ready(function(){
    $.get('assets/maps/map_one.xml', function (d) {


        var $map = $(d).find('map');
        var nodes = $(d).find('node');
//        var canvas=document.getElementById('map');
        var canvas = $("#map")[0];
        var ctx=canvas.getContext('2d');
        ctx.fillStyle='#FF0000';
        n = $(d).find("node").first();
//        ctx.fillRect(n.attr("x"), n.attr("y"),100,100);
//        ctx.strokeRect(240, 120, 100, 100);
//        nodes.each(function(){
//            var node = $(this);
//            node.attr("x", node.attr("x")*30-400);
//            node.attr("y", node.attr("y")*30-400);
//        });
        nodes.each(function(){
            var node = $(this);
            $("#node").append(node.attr("x"), ' ');
//            ctx.beginPath();
//            ctx.arc(70,18,15,0,Math.PI*2,true);
//            ctx.closePath();
            draw_rect(node.attr("x"),node.attr("y"));
        });
        $("#node").append('<br>')
        nodes.each(function(){
            var node = $(this);
            $("#node").append(node.attr("y"), ' ');
        });

        var edges = $(d).find("edge");
        edges.each(function(){

        });
//        alert(node.attr("y"));
        var $nodes = $map.find('nodes');
        $nodes.each(function() {
            var $node = $(this);
            var x = $node.attr("x");
            var item = $node.attr("item");
        });
        $('#show').click(function(){
            var canvas=document.getElementById('map');
            var ctx=canvas.getContext('2d');
            ctx.fillStyle='#FF0000';
            ctx.fillRect(0,0,40,60);
        });
        $(d).find('book').each(function () {
            var $book = $(this);
            var title = $book.attr("title");
            var description = $book.find('description').text();  //这里是读取字段内容
            var imageurl = $book.attr('imageurl');     //这里是读取字段属性值

            var html = '<dt> <img class="bookImage" alt="" src="' + imageurl + '" /> </dt>';
            html += '<dd> <span class="loadingPic" alt="Loading" />';
            html += '<p class="title">' + title + '</p>';
            html += '<p> ' + description + '</p>';
            html += '</dd>';
            $('dl').append($(html));
            $('.loadingPic').fadeOut(2000);
        });
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
