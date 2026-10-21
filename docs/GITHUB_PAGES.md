# Publish the study notebooks with GitHub Pages

The homepage is `index.html` in the project root. The five reading copies are in `output/notebooks/`. They already include saved outputs and embedded chart images. This is a static website; no Python environment or build command is needed to publish it.

## Use the existing project repository

1. Push the homepage, `.nojekyll`, and updated HTML reading copies to GitHub.
2. Open the repository's **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Choose branch **main** and folder **/(root)**, then **Save**.
5. Wait for GitHub's Pages deployment to finish. Open the site URL shown in those settings.

For this repository, the expected URL is:

`https://mitul-bhatia.github.io/t011-card-fraud-phase1/`

This is the expected address after publishing, not a claim that the site is already live.

The current project repository is private. GitHub Free supports Pages from public repositories; private repositories require a supported paid plan. This change does not alter repository visibility or enable Pages. You can instead use a separate public repository containing only the prepared website files.

Official reference: [GitHub Pages publishing-source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## Use a separate repository for the reading website

Extract the local `output/GitHub_Pages_Site.zip` and upload its **contents** to the new repository root. The ZIP contains:

```text
index.html
.nojekyll
output/
  Study_Notebook.html
  Phase1_Report.pdf
  Phase1_Presentation_Complete.pdf
  Phase1_Presentation_Complete.pptx
  notebooks/
    01_Project_and_Literature.html
    02_Data_Quality_and_EDA.html
    03_Feature_Engineering.html
    04_GMM_Training.html
    05_Evaluation_and_Thresholds.html
```

Keep these paths together. Enable Pages with **main / (root)** as above. The website ZIP contains the reading pages and linked documents; it excludes raw transactions, fitted models and the private build folder. The GitHub repository link on the homepage still points to the original project repository; access to that repository depends on its own permissions.

## Reading and updating

- Open `index.html` locally to preview the homepage.
- Readers can use **All notebooks**, **Previous notebook** and **Next notebook** at the top and bottom of each notebook.
- All site links use relative paths, so they work under a repository project URL.
- Notebook content and chart images are saved in the HTML. The original notebook exporter also loads external libraries for mathematical rendering; internet access helps display equations as intended.
- These pages show the saved run. Running the `.ipynb` files does not automatically update their HTML copies. After a changed experiment, export the reading copies again, retain their navigation, and update the report/slides to match before publishing.
