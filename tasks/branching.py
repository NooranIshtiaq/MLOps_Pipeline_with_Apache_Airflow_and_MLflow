def branch_model(ti):
    # Pull accuracy from XCom pushed by evaluate_model
    acc = ti.xcom_pull(task_ids="evaluate_model", key="accuracy")

    if acc is None:
        raise ValueError("Accuracy not found in XCom!")

    if acc >= 0.80:
        return "register_model"
    else:
        return "reject_model"