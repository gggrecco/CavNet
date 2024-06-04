document.getElementById('upload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const img = document.getElementById('image');
        img.src = URL.createObjectURL(file);
        img.style.display = 'block';

        const formData = new FormData();
        formData.append('file', file);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const charlieResponses = [
                "Whoa! That’s a charslee warslee",
                "That’s definitely a Karl!",
                "Well hello Sir Charles, King of the side of the Atlantic",
                "Clearly that’s a Bubbacita",
                "OMG is that a shark or a Charles, I can’t tell",
                "That boy Charlie is long enough to ride the dragster!",
                "That’s Charlie: leader of the 10th St gang"
            ];
            const rosieResponses = [
                "Whoa Beaner Weiner alert!",
                "That’s a Beanie Weinie if I’ve ever seen one!",
                "That’s a Rosie Bean",
                "Ahh Bean Lowein",
                "Its Rosie Posie!"
            ];

            let resultText = '';
            if (data.result === 'Charlie') {
                resultText = charlieResponses[Math.floor(Math.random() * charlieResponses.length)];
            } else if (data.result === 'Rosie') {
                resultText = rosieResponses[Math.floor(Math.random() * rosieResponses.length)];
            }

            document.getElementById('result').innerText = resultText;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});
