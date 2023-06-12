// Define variables
let movieData;
let currentPage = 1;
const rowsPerPage = 100;

// Function to fetch CSV file
async function fetchCSV() {
  const response = await fetch("movie_kmeans_2500.csv");
  const data = await response.text();
  return data;
}

// Function to parse CSV data into array of objects
function parseCSV(csv) {
  const lines = csv.split("\n");
  const headers = lines[0].split(",");
  const result = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const obj = {};
    const currentLine = lines[i].split(",");
    for (let j = 0; j < headers.length; j++) {
      obj[headers[j]] = currentLine[j];
    }
    result.push(obj);
    console.log(obj);
  }
  return result;
}

// Function to display movie data
function displayMovies(data, page) {
  const tableBody = document.querySelector("#movie-table tbody");
  tableBody.innerHTML = "";
  
  const startIndex = (page - 1) * rowsPerPage;
  const endIndex = startIndex + rowsPerPage;
  const displayedData = data.slice(startIndex, endIndex);
  
  displayedData.forEach(movie => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${movie.name}</td>
      <td>${movie.director}</td>
      <td>${movie.rating}</td>
      <td>${movie.category}</td>
      <td>${movie.duration}</td>
      <td>${movie.tag}</td>
      <td>${movie.lables}</td>
      <td><button onclick="showSimilar('${movie.lables}')">Show Similar</button></td>
    `;
    tableBody.appendChild(row);
  });
}

// Function to display pagination
function displayPagination(data) {
  const totalPages = Math.ceil(data.length / rowsPerPage);
  const paginationDiv = document.querySelector("#pagination");
  paginationDiv.innerHTML = "";
  
  for (let i = 1; i <= totalPages; i++) {
    const button = document.createElement("button");
    button.innerText = i;
    
    if (i === currentPage) {
      button.classList.add("active");
    }
    
    button.addEventListener("click", () => {
      currentPage = i;
      displayMovies(movieData, currentPage);
      displayPagination(movieData);
    });
    
    paginationDiv.appendChild(button);
  }
}

// Function to show movies with similar lables
function showSimilar(lables) {
  const similarMovies = movieData.filter(movie => movie.lables === lables);
  displayMovies(similarMovies, 1);
  displayPagination(similarMovies);
}

// Fetch and display CSV data
fetchCSV()
.then(data => {
  movieData = parseCSV(data);
  displayMovies(movieData, currentPage);
  displayPagination(movieData);
})
.catch(error => {
  console.log("Error fetching CSV file:", error);
});