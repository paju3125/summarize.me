(function($) {
    "use strict";

    $(window).on('load', function() {

        $('#userInputForm').on('submit', function(e) {
            e.preventDefault()
            $('#urlInputForm')[0].reset()
            document.getElementById('pegasus_summary').value = ''
            let btn = document.getElementById('genSummary')
            let btn2 = document.getElementById('genURLSummary')
            btn2.disabled = true
            btn2.form.firstElementChild.disabled = true;
            btn.classList.add('spin');
            btn.disabled = true;
            btn.form.firstElementChild.disabled = true;

            $('#btnTxt').text('Preprocessing')
            $.ajax({
                type: "post",
                url: "/preprocess",
                data: $('#userInputForm').serialize(),
                dataType: "json",
                success: function(response) {
                    console.log(response)
                    console.log(response.preprocessed_text)
                    $('#btnTxt').text('Generating Summary')
                    $.ajax({
                        type: "post",
                        url: "/generateSummary",
                        data: response,
                        // dataType: "dataType",
                        success: function(response) {
                            console.log(response)
                            console.log(response.summary)
                            document.getElementById('pegasus_summary').value = response.summary
                            $('#btnTxt').text('Checkout Summary')
                            setTimeout(() => {
                                $('#btnTxt').text('Generate Summary')
                                btn.disabled = false;
                                btn.form.firstElementChild.disabled = false;
                            }, 3000);
                            btn.classList.remove('spin');
                            btn2.disabled = false
                            btn2.form.firstElementChild.disabled = false;
                        }
                    });
                }
            });
        })

        // for url input
        $('#urlInputForm').on('submit', function(e) {
            e.preventDefault()
            $('#userInputForm')[0].reset()
            document.getElementById('pegasus_summary').value = ''
            let btn = document.getElementById('genSummary')
            let btn2 = document.getElementById('genURLSummary')
            btn.disabled = true
            btn.form.firstElementChild.disabled = true;
            btn2.classList.add('spin');
            btn2.disabled = true;
            btn2.form.firstElementChild.disabled = true;

            $('#btnTxt2').text('Preprocessing')
            $.ajax({
                type: "post",
                url: "/analyze_url",
                data: $('#urlInputForm').serialize(),
                success: function(response) {
                    console.log(response)
                    document.getElementById('message').value = response.rawText
                    $.ajax({
                        type: "post",
                        url: "/preprocess",
                        data: $('#userInputForm').serialize(),
                        dataType: "json",
                        success: function(response) {
                            console.log(response)
                            console.log(response.preprocessed_text)
                            $('#btnTxt2').text('Generating Summary')
                            $.ajax({
                                type: "post",
                                url: "/generateSummary",
                                data: response,
                                // dataType: "dataType",
                                success: function(response) {
                                    console.log(response)
                                    console.log(response.summary)
                                    document.getElementById('pegasus_summary').value = response.summary
                                    $('#btnTxt2').text('Checkout Summary')
                                    setTimeout(() => {
                                        $('#btnTxt').text('Generate Summary')
                                        btn2.disabled = false;
                                        btn2.form.firstElementChild.disabled = false;
                                    }, 3000);
                                    btn2.classList.remove('spin');
                                    btn.disabled = false
                                    btn.form.firstElementChild.disabled = false;
                                }
                            });
                        }
                    });

                }
            });
            // alert('from submitted')

        })


    })
})(jQuery);