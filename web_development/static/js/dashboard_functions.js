// Function to load the calendar table and the two line charts from the previous weeks
function load_table(){
  html = "<center><img src='/static/img/loading.gif' style = 'width: 50px; margin: 0 auto; margin: 100px;'> </img></center>"
  $('#calendar').html(html);
  $.getJSON($SCRIPT_ROOT + '/calendar_json', {
    orgcode: $('#orgcode').val(),
    modality: $('#modality').val(),
    departmentcode: $('#departmentcode').val()
  }, function(data) {
    columns_names = data.result.columns_names;
    row_names = data.result.row_names;
    rows = data.result.rows;
    ts_data = data.result.ts_data;
    ts_data2 = data.result.ts_data2;

    var html = "";
    html+='<table class="table table-bordered"><thead>';
    for (var i=0;i<columns_names.length;i++){
      html+= "<th>" + columns_names[i] + "</th>";
    }
    html+='</thead><tbody>';
    for (var i=0;i<rows.length;i++){
      html+= '<tr>';
      var row_name = row_names[i];
      html+= "<td>" + row_name + "</td>";

      var columns = rows[i];
      for (var j=0;j<columns.length;j++){
        var time_slot = columns[j];
        if (time_slot.status == 0){
          html+= "<td class='time_slot bg-grey' data-toggle='tooltip' data-container='body'"
          html+= "data-exam-id='"+time_slot.exam_id+"'"
          html+= "data-patient-id='"+time_slot.patient_id+"'"
          html+= "data-age='"+time_slot.age+"'"
          html+= "data-gender='"+time_slot.gender+"'"
          html+= "title = 'Cancel Probablity: "+time_slot.probability+"'></td>";
        } else if (time_slot.status == 1) {
          html+= "<td class='time_slot bg-blue' data-toggle='tooltip' data-container='body'"
          html+= "data-exam-id='"+time_slot.exam_id+"'"
          html+= "data-patient-id='"+time_slot.patient_id+"'"
          html+= "data-age='"+time_slot.age+"'"
          html+= "data-gender='"+time_slot.gender+"'"
          html+= "title = 'Cancel Probablity: "+time_slot.probability+"'></td>";
        } else {
          html+= "<td class='time_slot'></td>";
        }
      }
    html+= '</tr>';
    }
    html+='</tbody></table>';

    $('#calendar').html(html);
    $('[data-toggle="tooltip"]').tooltip();

    load_line_chart(ts_data);
    load_line_chart2(ts_data2);

  });
}

// Function to load the modalities list
function load_modalities(modalities){
  var $el = $("#modality");
  $el.empty(); // remove old options
  $.each(modalities, function(value, key) {
    $el.append($("<option></option>")
       .attr("value", key).text(key));
    });

}

// Function to load the departments list
function load_departments(departments){
  var $el = $("#departmentcode");
  $el.empty(); // remove old options
  $.each(departments, function(value, key) {
    $el.append($("<option></option>")
       .attr("value", key).text(key));
    });

}

// Function to reset the departments list to only Choose
function reset_departments(){
  $("#departmentcode").html('<option value="Choose">Choose</option>')
}

// Function to reset the modalities list to only Choose
function reset_modalities(){
  $("#modality").html('<option value="Choose">Choose</option>')
}

// Function to decorate text as information (blue color)
function text_info(text){
  return '<span class = "text-info">'+text+'</span>';
}

// Function to load patient information panel
function load_patient_info(patient_info){
  html = '<div class="panel panel-default">';
  html += '<div class="panel-heading">Patient Information</div>';
  html += '<div class="panel-body">';
  html += '<p>Name: '+text_info(patient_info.patient_id)+'</p>';
  html += '<p>Age: '+text_info(patient_info.age)+'</p>';
  html += '<p>Gender: '+text_info(patient_info.gender)+'</p>';
  html += '<p>Email: '+ text_info(patient_info.patient_id.slice(0,6)+"@gmail.com")+'</p>';
  //html += '<p>Phone number: '+text_info(patient_info.telephone)+'</p>';
  html += '</div></div></div>';
  $('#patient-info').html(html);
}

// Function to load the organizations chart
function load_org_chart(chart_data){
  chart_id = 'hospital-chart';
  html = '<canvas id="'+chart_id+'" width="600" height="600"></canvas>';
  $('#hospital-div').html(html);

  labels =chart_data.labels;
  data_show = chart_data.Show;
  data_noshow = chart_data.NoShow;
  title = 'Organization'
  create_stacked_chart(chart_id, labels, data_show, data_noshow, title)
}

// Function to load the modalities chart
function load_modalities_chart(chart_data){
  chart_id = 'modality-chart';
  html = '<canvas id="'+chart_id+'" width="600" height="600"></canvas>';
  $('#modality-div').html(html);

  labels =chart_data.labels;
  data_show = chart_data.Show;
  data_noshow = chart_data.NoShow;
  title = 'Modality'
  create_stacked_chart(chart_id, labels, data_show, data_noshow, title)
}

// Function to load the previous week chart
function load_line_chart(chart_data){
  chart_id = 'week-1';
  html = '<canvas id="'+chart_id+'" width="600" height="400"></canvas>';
  $('#weekm1').html(html);

  labels =chart_data.date_list;
  full_slots = chart_data.full_slots;
  slots_taken = chart_data.slots_taken;
  title = 'Slots Taken in Previous First Week'
  create_line_chart(chart_id, labels, full_slots, slots_taken, title)
}

// Function to load the second to last week chart
function load_line_chart2(chart_data){
  chart_id = 'week-2';
  html = '<canvas id="'+chart_id+'" width="600" height="400"></canvas>';
  $('#weekm2').html(html);

  labels =chart_data.date_list;
  full_slots = chart_data.full_slots;
  slots_taken = chart_data.slots_taken;
  title = 'Slots Taken in Previous Second Week'
  create_line_chart(chart_id, labels, full_slots, slots_taken, title)
}

// Function to  create a stacked chart
function create_stacked_chart(chart_id, labels, data_show, data_noshow, title){
     // stacked chart data
    var config = {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        type: 'bar',
        label: 'Show',
        backgroundColor: "#FF8800",
        data: data_show,
      }, {
        type: 'bar',
        label: "No show",
        backgroundColor: "#4285F4",
        data: data_noshow,
      }]
    },
    options: {
      title: {
            display: true,
            text: title
      },
      scales: {
        xAxes: [{
          stacked: true
        }],
        yAxes: [{
          stacked: true
        }]
      }
    }
  };
   // get stacked chart canvas
   var mychart = document.getElementById(chart_id).getContext("2d");
   new Chart(mychart, config);
}

// function to create an ovelaping line chart
function create_line_chart(chart_id, labels, full_slots, slots_taken, title){
     // line chart data
    var config = {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        type: 'line',
        label: 'Slots Taken',
        backgroundColor: "#ffccbc",
        data: slots_taken,
      }, {
        type: 'line',
        label: "Full Slots",
        backgroundColor: "#bbdefb",
        data: full_slots,
      }]
    },
    options: {
      title: {
            display: true,
            text: title
      },
      scales: {
        yAxes: [{
          ticks: {beginAtZero:true}
        }]
      }
    }
  };
   // get stacked chart canvas
   var mychart = document.getElementById(chart_id).getContext("2d");
   new Chart(mychart, config);
}
