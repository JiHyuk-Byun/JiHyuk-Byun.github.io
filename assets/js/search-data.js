const ninja = document.querySelector("ninja-keys");

ninja.data = [
  {
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },
  {
    id: "nav-publications",
    title: "publications",
    description: "Publications by JiHyuk Byun.",
    section: "Navigation",
    handler: () => {
      window.location.href = "/publications/";
    },
  },
  {
    id: "nav-projects",
    title: "projects",
    description: "Projects by JiHyuk Byun.",
    section: "Navigation",
    handler: () => {
      window.location.href = "/projects/";
    },
  },
  {
    id: "nav-cv",
    title: "CV",
    description: "CV of JiHyuk Byun.",
    section: "Navigation",
    handler: () => {
      window.location.href = "/cv/";
    },
  },
  {
    id: "pub-webstep",
    title: "Where Did It Go Wrong? Process-Level Evaluation of Web Agents with Semantic State Tracking",
    description: "Jiwan Chung, JiHyuk Byun, Vibhav Vineet, Seon Joo Kim. Published in April 2026.",
    section: "Publications",
    handler: () => {
      window.location.href = "/publications/";
    },
  },
  {
    id: "pub-3d-paqa",
    title: "Towards Preference-Aligned 3D Quality Assessment",
    description: "JiHyuk Byun and Seon Joo Kim. IPIU 2026.",
    section: "Publications",
    handler: () => {
      window.location.href = "/publications/";
    },
  },
  {
    id: "social-email",
    title: "email",
    section: "Socials",
    handler: () => {
      window.open("mailto:quswlgur123@yonsei.ac.kr", "_blank");
    },
  },
  {
    id: "social-scholar",
    title: "Google Scholar",
    section: "Socials",
    handler: () => {
      window.open("https://scholar.google.com/citations?user=ZUOimHsAAAAJ&hl=ko&oi=ao", "_blank");
    },
  },
  {
    id: "social-github",
    title: "GitHub",
    section: "Socials",
    handler: () => {
      window.open("https://github.com/JiHyuk-Byun", "_blank");
    },
  },
  {
    id: "social-linkedin",
    title: "LinkedIn",
    section: "Socials",
    handler: () => {
      window.open("https://www.linkedin.com/in/jihyuk-byun-b29545260/", "_blank");
    },
  },
];
