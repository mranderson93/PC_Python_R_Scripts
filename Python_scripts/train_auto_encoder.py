# Train autoencoder
def train_autoencoder(autoencoder, dataloader, criterion, optimizer, device = "cpu", epochs=20):
    autoencoder.to(device)
    autoencoder.train()
    LOSS = []
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            encoded, decoded = autoencoder(X_batch)
            loss = criterion(decoded, X_batch)
            LOSS.append(loss.item())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Autoencoder Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
        # if (epoch+1) % 20 == 0:
        #   print(f"Autoencoder Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
    return LOSS,autoencoder