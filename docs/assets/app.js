(function () {
  const page = document.body.dataset.page;
  document.querySelectorAll('.links a[data-page]').forEach((a) => {
    if (a.dataset.page === page) a.classList.add('active');
  });

  const sideLinks = Array.from(document.querySelectorAll('.side a'));
  if (!sideLinks.length) return;

  const sections = sideLinks
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        sideLinks.forEach((x) => x.classList.remove('current'));
        const active = sideLinks.find((x) => x.getAttribute('href') === `#${e.target.id}`);
        if (active) active.classList.add('current');
      });
    },
    { rootMargin: '-32% 0px -58% 0px', threshold: 0 },
  );

  sections.forEach((s) => io.observe(s));
})();
