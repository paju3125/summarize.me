(function($) {
    "use strict";

    $(window).on('load', function() {

        $('#userInputForm').on('submit', function(e) {
            e.preventDefault()

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
                    $('#btnTxt').text('Generating Summary')
                    $.ajax({
                        type: "post",
                        url: "/generateSummary",
                        data: response,
                        // dataType: "dataType",
                        success: function(response) {
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
            // alert('from submitted')

        })


    })
})(jQuery);