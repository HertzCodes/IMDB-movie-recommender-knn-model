from scripts import model, vectorizer, webscraper


def start_program():
    entry_title = input('Please enter your movie title: ')
    entry_summary = input('Please enter your movie summary: ')
    print('Do you want to scrape the movies datas? Y/N')
    choice = input('Enter your choice: ').lower()
    if choice == 'y':
        webscraper.start_process()
    vectorizer.start_process(entry_title, entry_summary)
    # sklearn_model.start_process(entry_title, entry_summary)
    model.knn_model(entry_title)


if __name__ == "__main__":
    start_program()
