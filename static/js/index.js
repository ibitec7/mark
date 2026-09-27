window.HELP_IMPROVE_VIDEOJS = false;

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');

    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';

            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);

            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (!scrollButton) return;
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');

    if (carouselVideos.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });

    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

// Keep every gallery advancing on its own, forever.
// The galleries are started at staggered offsets (one period divided across
// them), so no two galleries ever change slide at the same moment.
function startAutoAdvance(instances, periodMs) {
    var count = instances.length;

    instances.forEach(function (instance, i) {
        if (typeof instance.next !== 'function') return;

        var offset = Math.round((periodMs * (i + 1)) / count);

        window.setTimeout(function () {
            instance.next();
            window.setInterval(function () {
                instance.next();
            }, periodMs);
        }, offset);
    });
}

// Initialize the results carousels.
// bulma-carousel is self-contained (no jQuery needed) and is loaded before this file.
function initCarousels() {
    if (typeof window.bulmaCarousel === 'undefined') {
        console.warn('bulma-carousel unavailable; galleries are shown as a static list.');
        return;
    }

    var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        // The built-in autoplay is off: it would start every gallery on the same
        // tick. Advancing is driven below instead, on per-gallery timers.
        autoplay: false,
    };

    // Initialize all divs with the carousel class
    var galleries = window.bulmaCarousel.attach('.carousel', options);

    if (galleries && galleries.length) {
        startAutoAdvance(galleries, 15000);
    }

    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCarousels);
} else {
    initCarousels();
}
